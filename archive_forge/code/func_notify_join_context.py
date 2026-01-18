import warnings
from abc import ABC, abstractmethod
from types import TracebackType
from typing import Any, List, NamedTuple, Optional, Type
import torch
import torch.distributed as dist
@staticmethod
def notify_join_context(joinable: Joinable):
    """
        Notifies the join context manager that the calling process has not yet joined.

        Then, if ``throw_on_early_termination=True``, checks if uneven inputs have been detected
        (i.e. if one process has already joined) and throws an exception if so.

        This method should be called from a :class:`Joinable` object before
        its per-iteration collective communications. For example, this should
        be called at the beginning of the forward pass in
        :class:`DistributedDataParallel`.

        Only the first :class:`Joinable` object passed into the context
        manager performs the collective communications in this method, and
        for the others, this method is vacuous.

        Arguments:
            joinable (Joinable): the :class:`Joinable` object calling this
                method.

        Returns:
            An async work handle for the all-reduce meant to notify the context
            manager that the process has not yet joined if ``joinable`` is the
            first one passed into the context manager; ``None`` otherwise.
        """
    assert hasattr(joinable, '_join_config'), f'Check that the {type(joinable)} constructor calls the ``Joinable`` constructor'
    join_config = joinable._join_config
    if not join_config.is_first_joinable or not join_config.enable:
        return None
    device = joinable.join_device
    process_group = joinable.join_process_group
    ones = torch.ones(1, device=device)
    work = dist.all_reduce(ones, group=process_group, async_op=True)
    if join_config.throw_on_early_termination:
        zeros = torch.zeros(1, device=device)
        dist.all_reduce(zeros, group=process_group)
        should_throw = zeros.item()
        if should_throw:
            raise RuntimeError('Detected at least one rank that exhausted inputs. Throwing across all ranks.')
    return work