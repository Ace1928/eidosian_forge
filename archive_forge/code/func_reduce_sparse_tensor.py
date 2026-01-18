import multiprocessing
import os
import threading
from multiprocessing.reduction import ForkingPickler
from multiprocessing.util import register_after_fork
from typing import Union
import torch
import torch.utils.hooks
from torch._namedtensor_internals import check_serializing_named_tensor
def reduce_sparse_tensor(sparse):
    if sparse.layout is torch.sparse_coo:
        rebuild_indices_func, rebuild_indices_args = reduce_tensor(sparse._indices())
        rebuild_values_func, rebuild_values_args = reduce_tensor(sparse._values())
        return (rebuild_sparse_coo_tensor, (rebuild_indices_func, rebuild_indices_args, rebuild_values_func, rebuild_values_args, sparse.shape, sparse.is_coalesced()))
    else:
        if sparse.layout in {torch.sparse_csr, torch.sparse_bsr}:
            compressed_indices = sparse.crow_indices()
            plain_indices = sparse.col_indices()
        elif sparse.layout in {torch.sparse_csc, torch.sparse_bsc}:
            compressed_indices = sparse.ccol_indices()
            plain_indices = sparse.row_indices()
        else:
            raise NotImplementedError(sparse.layout)
        rebuild_compressed_indices_func, rebuild_compressed_indices_args = reduce_tensor(compressed_indices)
        rebuild_plain_indices_func, rebuild_plain_indices_args = reduce_tensor(plain_indices)
        rebuild_values_func, rebuild_values_args = reduce_tensor(sparse.values())
        return (rebuild_sparse_compressed_tensor, (rebuild_compressed_indices_func, rebuild_compressed_indices_args, rebuild_plain_indices_func, rebuild_plain_indices_args, rebuild_values_func, rebuild_values_args, sparse.shape, sparse.layout))