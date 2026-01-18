import multiprocessing
import os
import threading
from multiprocessing.reduction import ForkingPickler
from multiprocessing.util import register_after_fork
from typing import Union
import torch
import torch.utils.hooks
from torch._namedtensor_internals import check_serializing_named_tensor
def free_dead_references(self):
    live = 0
    for key, storage_ref in list(self.items()):
        if storage_ref.expired():
            del self[key]
        else:
            live += 1
    self.limit = max(128, live * 2)