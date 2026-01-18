import multiprocessing
import os
import threading
from multiprocessing.reduction import ForkingPickler
from multiprocessing.util import register_after_fork
from typing import Union
import torch
import torch.utils.hooks
from torch._namedtensor_internals import check_serializing_named_tensor
def reduce_nested_tensor(nt):
    rebuild_buffer_func, rebuild_buffer_args = reduce_tensor(nt.values())
    rebuild_sizes_func, rebuild_sizes_args = reduce_tensor(nt._nested_tensor_size())
    rebuild_strides_func, rebuild_strides_args = reduce_tensor(nt._nested_tensor_strides())
    rebuild_offsets_func, rebuild_offsets_args = reduce_tensor(nt._nested_tensor_storage_offsets())
    return (rebuild_nested_tensor, (rebuild_buffer_func, rebuild_buffer_args, rebuild_sizes_func, rebuild_sizes_args, rebuild_strides_func, rebuild_strides_args, rebuild_offsets_func, rebuild_offsets_args))