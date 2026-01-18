from typing import List
import torch
from . import config, ir, scheduler
from .dependencies import WeakDep
from .utils import tuple_sorted
def sink_waits(snodes: List['scheduler.BaseSchedulerNode']) -> List['scheduler.BaseSchedulerNode']:
    """
    Greedily moves waits as late as possible (i.e. until we reach a use). Optimal in terms of
    communication overlap.
    """
    new_order = []
    cur_waits = set()
    for snode in snodes:
        if isinstance(snode.node, ir.Wait):
            cur_waits.add(snode)
        else:
            for wait in tuple_sorted(cur_waits):
                if snode in wait.node_users:
                    new_order.append(wait)
                    cur_waits.remove(wait)
            new_order.append(snode)
    for snode in tuple_sorted(cur_waits):
        new_order.append(snode)
    return new_order