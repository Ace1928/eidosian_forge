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
def checkTrace(self, func, reference_tensors, input_tensors=None, drop=None, allow_unused=False, verbose=False, inputs_require_grads=True, check_tolerance=1e-05, export_import=True, _force_outplace=False, grad_atol=None, grad_rtol=None):

    def allSum(vs):
        if drop is not None:
            vs = vs[:-drop]
        return sum((math.log(i + 2) * v.sum() for i, v in enumerate(vs) if v is not None))
    if input_tensors is None:
        input_tensors = reference_tensors

    def flatten_inputs(inputs):

        def input_reduce(input, fn, acc):
            if isinstance(input, torch.Tensor):
                fn(input, acc)
            elif isinstance(input, dict):
                reduce(lambda acc, key: input_reduce(input[key], fn, acc), input, acc)
            else:
                reduce(lambda acc, val: input_reduce(val, fn, acc), input, acc)
            return acc
        return tuple(input_reduce(recording_inputs, lambda t, acc: acc.append(t), []))
    nograd_inputs = reference_tensors
    if inputs_require_grads:
        recording_inputs = do_input_map(lambda t: t.clone().requires_grad_(), reference_tensors)
        flattened_recording_inputs = flatten_inputs(recording_inputs)
    else:
        recording_inputs = reference_tensors
    ge = torch.jit.trace(func, input_tensors, check_tolerance=check_tolerance, _force_outplace=_force_outplace, check_trace=False)
    if export_import:
        ge = self.getExportImportCopy(ge)
    if verbose:
        print(ge.graph)
    outputs = func(*nograd_inputs)
    outputs_ge = ge(*nograd_inputs)
    self.assertEqual(outputs, outputs_ge)
    outputs = func(*recording_inputs)
    if inputs_require_grads:
        grads = torch.autograd.grad(allSum(outputs), flattened_recording_inputs, allow_unused=allow_unused)
    outputs_ge = ge(*recording_inputs)
    if inputs_require_grads:
        grads_ge = torch.autograd.grad(allSum(outputs_ge), flattened_recording_inputs, allow_unused=allow_unused)
    self.assertEqual(outputs, outputs_ge)
    if inputs_require_grads:
        self.assertEqual(grads, grads_ge, atol=grad_atol, rtol=grad_rtol)
    outputs = func(*recording_inputs)
    l1 = allSum(outputs)
    if inputs_require_grads:
        grads = torch.autograd.grad(l1, flattened_recording_inputs, create_graph=True, allow_unused=allow_unused)
    if inputs_require_grads:
        l2 = allSum(grads) * l1
        grads2 = torch.autograd.grad(l2, flattened_recording_inputs, allow_unused=allow_unused)
    if inputs_require_grads:
        recording_inputs = do_input_map(lambda t: Variable(t, requires_grad=True), reference_tensors)
        flattened_recording_inputs = flatten_inputs(recording_inputs)
    outputs_ge = ge(*recording_inputs)
    l1_ge = allSum(outputs_ge)
    if inputs_require_grads:
        grads_ge = torch.autograd.grad(l1_ge, flattened_recording_inputs, create_graph=True, allow_unused=allow_unused)
    if inputs_require_grads:
        l2_ge = allSum(grads_ge) * l1_ge
        grads2_ge = torch.autograd.grad(l2_ge, flattened_recording_inputs, allow_unused=allow_unused)
    self.assertEqual(outputs, outputs_ge)
    if inputs_require_grads:
        self.assertEqual(grads, grads_ge, atol=grad_atol, rtol=grad_rtol)
        for g2, g2_ge in zip(grads2, grads2_ge):
            if g2 is None and g2_ge is None:
                continue
            self.assertEqual(g2, g2_ge, atol=0.0008, rtol=0.0008)
    return ge