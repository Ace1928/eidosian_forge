import argparse
import contextlib
import copy
import ctypes
import errno
import functools
import gc
import inspect
import io
import json
import logging
import math
import operator
import os
import platform
import random
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
import warnings
from collections.abc import Mapping, Sequence
from contextlib import closing, contextmanager
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from functools import partial, wraps
from itertools import product, chain
from pathlib import Path
from statistics import mean
from typing import (
from unittest.mock import MagicMock
import expecttest
import numpy as np
import __main__  # type: ignore[import]
import torch
import torch.backends.cudnn
import torch.backends.mkl
import torch.backends.mps
import torch.backends.xnnpack
import torch.cuda
from torch import Tensor
from torch._C import ScriptDict, ScriptList  # type: ignore[attr-defined]
from torch._utils_internal import get_writable_path
from torch.nn import (
from torch.onnx import (
from torch.testing import make_tensor
from torch.testing._comparison import (
from torch.testing._comparison import not_close_error_metas
from torch.testing._internal.common_dtype import get_all_dtypes
import torch.utils._pytree as pytree
from .composite_compliance import no_dispatch
def generate_simple_inputs(self, layout, device=None, dtype=None, index_dtype=None, enable_batch=True, enable_hybrid=True, enable_zero_sized=True, enable_non_contiguous_indices=True, enable_non_contiguous_values=True, enable_batch_variable_nse=False, output_tensor=True, patterns=None):
    """Generator of simple inputs for tensor constructors of the given layout.

        The generated tensor inputs have the following properties:

        - tensor shapes are minimal but not trivial
        - tensor values are sorted sequences for COO and CSR formats, e.g. [1, 2, 3, 4]
        - the generated tensors represent the same mathematical tensor for all layouts
        - the generated tensors include regular, zero-sized, and optionally, batched or/and hybrid tensors.
        - the generated tensors include contiguous or non-contiguous tensors both in indices and values

        If output_tensor is True, yield tensors with the given
        layout. Otherwise, yield inputs to the corresponding tensor
        constructors:

          - sparse compressed input is defined as
            (compressed_indices, plain_indices, values), dict(size=expected_size_from_shape_inference, device=device, dtype=dtype)

          - sparse COO input is defined as
            (indices, values), dict(size=expected_size_from_shape_inference, device=device, dtype=dtype)

          - strided input is defined as
            (values,), dict(device=device, dtype=dtype)
        """
    if index_dtype is None:
        index_dtype = torch.int64
    is_compressed_sparse_layout = layout in {torch.sparse_csr, torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc}
    if output_tensor:
        for args, kwargs in self.generate_simple_inputs(layout, device=device, dtype=dtype, index_dtype=index_dtype, enable_batch=enable_batch, enable_hybrid=enable_hybrid, enable_zero_sized=enable_zero_sized, enable_non_contiguous_indices=enable_non_contiguous_indices, enable_non_contiguous_values=enable_non_contiguous_values, enable_batch_variable_nse=enable_batch_variable_nse, output_tensor=False):
            if layout is torch.strided:
                assert len(args) == 1
                size = kwargs.pop('size', None)
                assert size is not None
                yield args[0].reshape(size)
            elif layout is torch.sparse_coo:
                yield torch.sparse_coo_tensor(*args, **kwargs)
            elif is_compressed_sparse_layout:
                kwargs.update(layout=layout)
                yield torch.sparse_compressed_tensor(*args, **kwargs)
            else:
                assert 0
        return

    def get_blockpattern(pattern, blocksize):
        basesize = pattern.shape
        assert basesize[0] % blocksize[0] == 0, (basesize, blocksize)
        assert basesize[1] % blocksize[1] == 0, (basesize, blocksize)
        blockpattern = pattern.reshape(-1, blocksize[0], basesize[1] // blocksize[1], blocksize[1]).transpose(-3, -2).any(-1).any(-1)
        block_ids = torch.arange(1, blockpattern.numel() + 1).reshape(blockpattern.shape)
        return (blockpattern != 0) * block_ids

    def get_sparse_data(pattern):
        basesize = pattern.shape
        assert len(basesize) == 2, basesize
        indices = torch.where(pattern != 0)
        coo_indices = torch.stack(indices)
        crow_indices = torch.zeros(basesize[0] + 1, dtype=torch.int64)
        crow_indices[1:] = torch.cumsum(coo_indices[0].bincount(minlength=basesize[0]), 0)
        col_indices = coo_indices[1]
        strided_values = torch.zeros(basesize, dtype=torch.int64)
        values = torch.arange(1, 1 + len(indices[0]), dtype=torch.int64)
        strided_values[indices] = values
        indices_T = torch.where(pattern.transpose(0, 1) != 0)
        coo_indices_T = torch.stack(indices_T)
        ccol_indices = torch.zeros(basesize[1] + 1, dtype=torch.int64)
        ccol_indices[1:] = torch.cumsum(coo_indices_T[0].bincount(minlength=basesize[1]), 0)
        row_indices = coo_indices_T[1]
        csc_values = strided_values.transpose(0, 1)[indices_T]
        return {torch.sparse_coo: (coo_indices, values), torch.sparse_csr: (crow_indices, col_indices, values), torch.sparse_csc: (ccol_indices, row_indices, csc_values), torch.strided: (strided_values,)}

    def get_sparse_data_with_block(pattern, blocksize):
        nonblock_data = get_sparse_data(pattern)
        blockpattern = get_blockpattern(pattern, blocksize)
        block_data = get_sparse_data(blockpattern)
        strided_values = nonblock_data[torch.strided][0]
        block_indices = block_data[torch.sparse_coo][0]
        bsr_values = torch.stack([strided_values[bi * blocksize[0]:(bi + 1) * blocksize[0], bj * blocksize[1]:(bj + 1) * blocksize[1]] for bi, bj in block_indices.transpose(0, 1)])
        bsc_values = bsr_values[block_data[torch.sparse_csc][2] - 1]
        return {torch.sparse_bsr: (*block_data[torch.sparse_csr][:2], bsr_values), torch.sparse_bsc: (*block_data[torch.sparse_csc][:2], bsc_values), **nonblock_data}

    def get_batch_sparse_data(pattern, blocksize):
        size = pattern.shape
        if len(size) <= 2:
            return get_sparse_data_with_block(pattern, blocksize)
        batch_data = {}
        for i, item in enumerate(pattern):
            for layout, d in get_batch_sparse_data(item, blocksize).items():
                target = batch_data.get(layout)
                if layout is torch.sparse_coo:
                    ext_coo_indices1 = torch.cat((torch.full((1, len(d[1])), i, dtype=torch.int64), d[0]))
                    if target is None:
                        target = batch_data[layout] = (ext_coo_indices1, d[1])
                    else:
                        target[0].set_(torch.cat((target[0], ext_coo_indices1), 1))
                        target[1].set_(torch.cat((target[1], d[1])))
                elif target is None:
                    target = batch_data[layout] = tuple((d[j].unsqueeze(0) for j in range(len(d))))
                else:
                    for j in range(len(d)):
                        target[j].set_(torch.cat((target[j], d[j].unsqueeze(0))))
        return batch_data

    def generate_values(base, densesize):
        """Generates a tensor of shape densesize with values equal to

              base + i_1 * 10^0 + ... + i_d * 10^{d - 1}

            at indices i_1, ..., i_d (with 0 <= i_j < densesize[j] for any 1 <= j <=
            len(densesize))

            This mapping produces unique values as long as
            densesize[i] < 10 for all i in range(len(densesize)).
            """
        if not densesize:
            return base
        if not isinstance(base, int) and base.ndim > 0:
            return torch.stack([generate_values(b, densesize) for b in base])
        if base == 0:
            return torch.zeros(densesize, dtype=torch.int64)
        r = torch.arange(densesize[0], dtype=torch.int64)
        for i, d in enumerate(densesize[1:]):
            y = torch.arange(d, dtype=torch.int64) * 10 ** (i + 1)
            r = r[..., None] + y[None, ...]
        r.add_(base)
        return r
    if patterns is None:
        patterns = [([[1, 2, 0], [1, 0, 3]], [(2, 1), (1, 3)], [(), (2,), (4, 5)]), ([[[[1, 2, 0], [1, 0, 3]], [[1, 2, 3], [1, 0, 0]], [[1, 0, 0], [1, 2, 3]]], [[[0, 2, 0], [1, 2, 3]], [[1, 0, 3], [1, 2, 0]], [[1, 2, 3], [0, 2, 0]]]], [(2, 1), (2, 3)], [(), (2,)]), ([[0, 1, 0, 2, 0, 2], [0, 1, 0, 0, 2, 0], [3, 3, 3, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 5, 0, 6, 6, 6], [5, 0, 5, 6, 6, 6], [0, 0, 0, 0, 8, 8], [7, 7, 7, 0, 8, 8]], [(2, 3)], [(), (4, 5)]), ([[[1, 2], [3, 4]], [[1, 0], [0, 0]]], [(1, 1)], [()] if enable_batch_variable_nse else [])]

    def non_contiguous_copy(t, dim=-1, offset=0):
        self.assertTrue(t.is_contiguous())
        if dim < 0:
            dim = dim + t.ndim
        assert dim >= 0 and dim < t.ndim
        step = max(2, offset + 1)
        tmp = torch.zeros((*t.shape[:dim], t.shape[dim] * step, *t.shape[dim + 1:]), dtype=t.dtype, device=t.device)
        dim_slices = (*(slice(None),) * dim, slice(offset, None, step))
        r = tmp[dim_slices].copy_(t)
        self.assertFalse(r.is_contiguous())
        self.assertEqual(t, r)
        return r
    for pattern, blocksizes, densesizes in patterns:
        if not enable_hybrid:
            densesizes = [s for s in densesizes if not s]
        if not (densesizes and blocksizes):
            continue
        pattern = torch.tensor(pattern, dtype=torch.int64)
        if not enable_batch and pattern.ndim > 2:
            continue
        for blocksize in blocksizes:
            data = get_batch_sparse_data(pattern, blocksize)[layout]
            for densesize in densesizes:
                indices = [a.to(device=device, dtype=index_dtype) for a in data[:-1]]
                values = generate_values(data[-1], densesize).to(device=device, dtype=dtype)
                yield ((*indices, values), dict(device=device, dtype=dtype, size=pattern.shape + densesize))
                if enable_non_contiguous_indices and pattern.ndim > 2:
                    for dim, offset in {(0, 1), (-2, 0)}:
                        indices_copy = [non_contiguous_copy(a, dim=dim, offset=offset) for a in indices]
                        yield ((*indices_copy, values), dict(device=device, dtype=dtype, size=pattern.shape + densesize))
                        if enable_non_contiguous_values:
                            values_copy = non_contiguous_copy(values, dim=-1, offset=1)
                            yield ((*indices_copy, values_copy), dict(device=device, dtype=dtype, size=pattern.shape + densesize))
                if enable_non_contiguous_values:
                    values_copy = non_contiguous_copy(values, dim=-1, offset=1)
                    yield ((*indices, values_copy), dict(device=device, dtype=dtype, size=pattern.shape + densesize))
    if enable_zero_sized:
        for basesize, blocksizes, densesizes in [((2, 0), [(1, 2)], [(), (2,), (2, 3)] if enable_hybrid else [()]), ((0, 2), [(1, 2), (2, 1), (3, 2)], [()]), ((0, 0), [(1, 2)], [()])]:
            for blocksize in blocksizes:
                for densesize in densesizes:
                    if layout == torch.strided:
                        indices = ()
                        values = torch.empty(basesize + densesize, device=device, dtype=dtype)
                    elif layout == torch.sparse_coo:
                        indices = (torch.empty(len(basesize), 0, device=device, dtype=index_dtype),)
                        values = torch.empty((0, *densesize), device=device, dtype=dtype)
                    elif layout == torch.sparse_csr:
                        crow_indices = torch.tensor([0] * (basesize[0] + 1), device=device, dtype=index_dtype)
                        col_indices = torch.empty(0, device=device, dtype=index_dtype)
                        indices = (crow_indices, col_indices)
                        values = torch.empty((0, *densesize), device=device, dtype=dtype)
                    elif layout == torch.sparse_csc:
                        ccol_indices = torch.tensor([0] * (basesize[1] + 1), device=device, dtype=index_dtype)
                        row_indices = torch.empty(0, device=device, dtype=index_dtype)
                        indices = (ccol_indices, row_indices)
                        values = torch.empty((0, *densesize), device=device, dtype=dtype)
                    elif layout == torch.sparse_bsr:
                        crow_indices = torch.tensor([0] * (basesize[0] // blocksize[0] + 1), device=device, dtype=index_dtype)
                        col_indices = torch.empty(0, device=device, dtype=index_dtype)
                        indices = (crow_indices, col_indices)
                        values = torch.empty((0, *blocksize, *densesize), device=device, dtype=dtype)
                    elif layout == torch.sparse_bsc:
                        ccol_indices = torch.tensor([0] * (basesize[1] // blocksize[1] + 1), device=device, dtype=index_dtype)
                        row_indices = torch.empty(0, device=device, dtype=index_dtype)
                        indices = (ccol_indices, row_indices)
                        values = torch.empty((0, *blocksize, *densesize), device=device, dtype=dtype)
                    else:
                        assert 0
                    yield ((*indices, values), dict(device=device, dtype=dtype, size=basesize + densesize))