import multiprocessing
import os
import threading
from multiprocessing.reduction import ForkingPickler
from multiprocessing.util import register_after_fork
from typing import Union
import torch
import torch.utils.hooks
from torch._namedtensor_internals import check_serializing_named_tensor
def rebuild_sparse_compressed_tensor(rebuild_compressed_indices_func, rebuild_compressed_indices_args, rebuild_plain_indices_func, rebuild_plain_indices_args, rebuild_values_func, rebuild_values_args, shape, layout):
    compressed_indices = rebuild_compressed_indices_func(*rebuild_compressed_indices_args)
    plain_indices = rebuild_plain_indices_func(*rebuild_plain_indices_args)
    values = rebuild_values_func(*rebuild_values_args)
    return torch.sparse_compressed_tensor(compressed_indices, plain_indices, values, shape, layout=layout)