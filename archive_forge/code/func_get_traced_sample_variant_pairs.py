from torch.autograd import Variable
from torch.autograd.function import _nested_map
from torch.jit.annotations import BroadcastingList2, BroadcastingList3  # noqa: F401
from torch.onnx import OperatorExportTypes
import torch
import torch.cuda
import torch.jit
import torch.jit._logging
import torch.jit.frontend
import torch.jit.quantized
import zipfile
import functools
from torch.testing import FileCheck
from torch.testing._internal.common_utils import IS_WINDOWS, \
from torch.testing._internal.common_jit import JitCommonTestCase
from torch.testing._internal.common_utils import enable_profiling_mode  # noqa: F401
from contextlib import contextmanager
from functools import reduce
from io import StringIO
from collections import defaultdict
import importlib.util
import inspect
import io
import math
import os
import pickle
import sys
import tempfile
import textwrap
from importlib.abc import Loader
from typing import Any, Dict, List, Tuple, Union
def get_traced_sample_variant_pairs(device, dtype, op):
    outputs: List[Tuple[Any, Any]] = []
    samples = op.sample_inputs(device, dtype)
    func = op.get_op()
    method = op.get_method()
    variants = {'function': func, 'method': method}
    has_fake_function = op.name in ['resize_', 'resize_as_']
    if has_fake_function:
        variants = {'method': getattr(torch.Tensor, op.name)}
    ops_with_unsupported_bool_args = [{'name': 'div_floor_rounding', 'arg_idx': [0]}, {'name': 'div_no_rounding_mode', 'arg_idx': [0]}, {'name': 'div_trunc_rounding', 'arg_idx': [0]}, {'name': 'index_fill', 'arg_idx': [2]}, {'name': 'full_like', 'arg_idx': [0]}, {'name': 'mul', 'arg_idx': [0]}, {'name': 'new_full', 'arg_idx': [1]}]
    if has_fake_function:
        return outputs
    for sample in samples:
        for variant in variants.values():
            if variant is None:
                continue
            if is_lambda(variant):
                continue
            matching_ops = filter(lambda x: op.formatted_name == x['name'], ops_with_unsupported_bool_args)
            for op_data in matching_ops:
                for idx in op_data['arg_idx']:
                    args = list(sample.args)
                    if len(sample.args) > idx and isinstance(sample.args[idx], bool):
                        args[idx] = int(args[idx])
                    sample.args = tuple(args)
            outputs.append((variant, sample))
    return outputs