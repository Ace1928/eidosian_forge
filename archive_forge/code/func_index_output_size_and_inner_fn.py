import functools
import itertools
import logging
import os
import warnings
from collections import defaultdict
from collections.abc import Iterable
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import sympy
import torch
import torch.fx
import torch.utils._pytree as pytree
from torch._higher_order_ops.triton_kernel_wrap import (
from torch._prims_common import (
from torch.fx.experimental.sym_node import magic_methods, method_to_operator
from torch.utils._sympy.functions import CeilDiv, FloorDiv, ModularIndexing
from .._dynamo.utils import import_submodule
from . import config, inductor_prims, ir, test_operators  # NOQA: F401
from .decomposition import decompositions, get_decompositions
from .ir import (
from .utils import (
from .virtualized import ops, V
from . import kernel
import_submodule(kernel)
from . import quantized_lowerings
def index_output_size_and_inner_fn(x_size, indices, tensor_indices, tensor_size, indices_loaders, indexed_size, x_loader, check):
    non_consecutive_tensors = False
    for previous, current in zip(tensor_indices, tensor_indices[1:]):
        if current - previous != 1:
            non_consecutive_tensors = True
    output_size = [x_size[i] for i, val in enumerate(indices) if val is None]
    output_size = [*output_size, *x_size[len(output_size) + len(tensor_indices):]]
    first_tensor_index = tensor_indices[0]
    if non_consecutive_tensors:
        output_size = tensor_size + output_size
    else:
        output_size = output_size[:first_tensor_index] + tensor_size + output_size[first_tensor_index:]

    def fn(idx):
        assert len(idx) == len(output_size)
        assert len(indices_loaders) == len(indexed_size)
        rank = len(tensor_size)
        new_index = []
        first_tensor_index = tensor_indices[0]
        start_offset = 0 if non_consecutive_tensors else first_tensor_index
        next_idx = 0
        for i in range(tensor_indices[-1] + 1):
            if i == start_offset:
                next_idx += rank
            if indices[i] is None:
                assert next_idx < len(idx)
                new_index.append(idx[next_idx])
                next_idx += 1
            else:
                loader = indices_loaders[i]
                assert loader is not None
                size = indexed_size[i]
                new_index.append(ops.indirect_indexing(loader(idx[start_offset:start_offset + rank]), size, check=check))
        new_index = [*new_index, *idx[next_idx:]]
        return new_index if x_loader is None else x_loader(new_index)
    return (output_size, fn)