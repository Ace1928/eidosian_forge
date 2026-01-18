import multiprocessing
import os
import threading
from multiprocessing.reduction import ForkingPickler
from multiprocessing.util import register_after_fork
from typing import Union
import torch
import torch.utils.hooks
from torch._namedtensor_internals import check_serializing_named_tensor
def rebuild_sparse_coo_tensor(rebuild_indices_func, rebuild_indices_args, rebuild_values_func, rebuild_values_args, shape, is_coalesced):
    indices = rebuild_indices_func(*rebuild_indices_args)
    values = rebuild_values_func(*rebuild_values_args)
    return torch.sparse_coo_tensor(indices, values, shape, is_coalesced=is_coalesced)