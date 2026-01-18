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
def maybe_unroll_group(group):
    """Unroll group with only one member.
    This allows treating a group of a single task as if it
    was a single task without pre-knowledge."""
    try:
        size = len(group.tasks)
    except TypeError:
        try:
            size = group.tasks.__length_hint__()
        except (AttributeError, TypeError):
            return group
        else:
            return list(group.tasks)[0] if size == 1 else group
    else:
        return group.tasks[0] if size == 1 else group