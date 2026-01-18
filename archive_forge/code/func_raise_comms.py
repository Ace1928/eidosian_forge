from typing import List
import torch
from . import config, ir, scheduler
from .dependencies import WeakDep
from .utils import tuple_sorted
def raise_comms(snodes: List['scheduler.BaseSchedulerNode']) -> List['scheduler.BaseSchedulerNode']:
    """
    Greedily moves comms as early as possible (i.e. until we reach an input).
    Optimal in terms of communication overlap.

    TODO: We might want to adjust this in the future to account for memory limitations.
    e.g. when we are compiling FSDP, this heuristics will cause the all-gathers to be prefetched as soon as possible,
    which is the beginning of the forwards pass. We'll have to either do a special pass for FSDP,
    or we'll want to redo this pass with memory considerations so we handle the FSDP case in a general way.
    """
    new_order_reversed: List['scheduler.BaseSchedulerNode'] = []
    cur_comms: List['scheduler.BaseSchedulerNode'] = []
    for snode in reversed(snodes):
        if isinstance(snode.node, ir.CollectiveKernel):
            cur_comms.append(snode)
        else:
            for comm in cur_comms:
                assert len(comm.inverse_users) > 0
            while len(cur_comms) > 0 and any((snode in comm.inverse_users for comm in cur_comms)):
                comm = cur_comms.pop(0)
                new_order_reversed.append(comm)
            new_order_reversed.append(snode)
    assert len(cur_comms) <= 1
    for snode in tuple_sorted(cur_comms):
        new_order_reversed.append(snode)
    return new_order_reversed[::-1]