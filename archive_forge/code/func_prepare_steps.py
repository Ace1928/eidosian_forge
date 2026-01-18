import itertools
import operator
import warnings
from abc import ABCMeta, abstractmethod
from collections import deque
from collections.abc import MutableSequence
from copy import deepcopy
from functools import partial as _partial
from functools import reduce
from operator import itemgetter
from types import GeneratorType
from kombu.utils.functional import fxrange, reprcall
from kombu.utils.objects import cached_property
from kombu.utils.uuid import uuid
from vine import barrier
from celery._state import current_app
from celery.exceptions import CPendingDeprecationWarning
from celery.result import GroupResult, allow_join_result
from celery.utils import abstract
from celery.utils.collections import ChainMap
from celery.utils.functional import _regen
from celery.utils.functional import chunks as _chunks
from celery.utils.functional import is_list, maybe_list, regen, seq_concat_item, seq_concat_seq
from celery.utils.objects import getitem_property
from celery.utils.text import remove_repeating_from_task, truncate
def prepare_steps(self, args, kwargs, tasks, root_id=None, parent_id=None, link_error=None, app=None, last_task_id=None, group_id=None, chord_body=None, clone=True, from_dict=Signature.from_dict, group_index=None):
    """Prepare the chain for execution.

        To execute a chain, we first need to unpack it correctly.
        During the unpacking, we might encounter other chains, groups, or chords
        which we need to unpack as well.

        For example:
        chain(signature1, chain(signature2, signature3)) --> Upgrades to chain(signature1, signature2, signature3)
        chain(group(signature1, signature2), signature3) --> Upgrades to chord([signature1, signature2], signature3)

        The responsibility of this method is to ensure that the chain is
        correctly unpacked, and then the correct callbacks are set up along the way.

        Arguments:
            args (Tuple): Partial args to be prepended to the existing args.
            kwargs (Dict): Partial kwargs to be merged with existing kwargs.
            tasks (List[Signature]): The tasks of the chain.
            root_id (str): The id of the root task.
            parent_id (str): The id of the parent task.
            link_error (Union[List[Signature], Signature]): The error callback.
                will be set for all tasks in the chain.
            app (Celery): The Celery app instance.
            last_task_id (str): The id of the last task in the chain.
            group_id (str): The id of the group that the chain is a part of.
            chord_body (Signature): The body of the chord, used to synchronize with the chain's
                last task and the chord's body when used together.
            clone (bool): Whether to clone the chain's tasks before modifying them.
            from_dict (Callable): A function that takes a dict and returns a Signature.

        Returns:
            Tuple[List[Signature], List[AsyncResult]]: The frozen tasks of the chain, and the async results
        """
    app = app or self.app
    use_link = self._use_link
    if use_link is None and app.conf.task_protocol == 1:
        use_link = True
    steps = deque(tasks)
    steps_pop = steps.pop
    steps_extend = steps.extend
    prev_task = None
    prev_res = None
    tasks, results = ([], [])
    i = 0
    while steps:
        task = steps_pop()
        is_first_task, is_last_task = (not steps, not i)
        if not isinstance(task, abstract.CallableSignature):
            task = from_dict(task, app=app)
        if isinstance(task, group):
            task = maybe_unroll_group(task)
        if clone:
            if is_first_task:
                task = task.clone(args, kwargs)
            else:
                task = task.clone()
        elif is_first_task:
            task.args = tuple(args) + tuple(task.args)
        if isinstance(task, _chain):
            steps_extend(task.tasks)
            continue
        if isinstance(task, group) and prev_task:
            tasks.pop()
            results.pop()
            try:
                task = chord(task, body=prev_task, task_id=prev_res.task_id, root_id=root_id, app=app)
            except AttributeError:
                task = chord(task, body=prev_task, root_id=root_id, app=app)
            if tasks:
                prev_task = tasks[-1]
                prev_res = results[-1]
            else:
                prev_task = None
                prev_res = None
        if is_last_task:
            res = task.freeze(last_task_id, root_id=root_id, group_id=group_id, chord=chord_body, group_index=group_index)
        else:
            res = task.freeze(root_id=root_id)
        i += 1
        if prev_task:
            if use_link:
                task.link(prev_task)
            if prev_res and (not prev_res.parent):
                prev_res.parent = res
        if link_error:
            for errback in maybe_list(link_error):
                task.link_error(errback)
        tasks.append(task)
        results.append(res)
        prev_task, prev_res = (task, res)
        if isinstance(task, chord):
            app.backend.ensure_chords_allowed()
            node = res
            while node.parent:
                node = node.parent
            prev_res = node
    self.id = last_task_id
    return (tasks, results)