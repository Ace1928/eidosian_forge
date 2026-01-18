from typing import List
import torch
from . import config, ir, scheduler
from .dependencies import WeakDep
from .utils import tuple_sorted
def reorder_compute_and_comm_for_overlap(snodes: List['scheduler.BaseSchedulerNode']) -> List['scheduler.BaseSchedulerNode']:
    order = snodes
    for p in config.reorder_for_compute_comm_overlap_passes:
        if isinstance(p, str) and p in globals():
            p = globals()[p]
        if torch.distributed.get_rank() == 0:
            overlap_log.debug(f'==== Visualize overlap before reordering pass {p} ====')
            try:
                visualize_overlap(order)
            except Exception as e:
                overlap_log.debug(str(e))
        order = p(order)
        if torch.distributed.get_rank() == 0:
            overlap_log.debug(f'==== Visualize overlap after reordering pass {p} ====')
            try:
                visualize_overlap(order)
            except Exception as e:
                overlap_log.debug(str(e))
    return order