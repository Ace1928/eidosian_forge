from __future__ import annotations
import functools
import inspect
import sys
import typing
import warnings
from typing import (
import torch
import torch._C._onnx as _C_onnx
from torch import _C
from torch.onnx import _constants, _type_utils, errors
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import _beartype, jit_utils
from torch.types import Number
@_beartype.beartype
def quantized_args(*arg_q_descriptors: bool, scale: Optional[float]=None, zero_point: Optional[int]=None, quantize_output: bool=True):
    """A decorator which extends support for quantized version of the base operator.

    Quantization is detected by examining the arguments that are annotated by
    `arg_q_descriptors`.

    If quantization is detected, the base operator symbolic function will be wrapped with
    argument de-quantization and output quantization.

    Otherwise, only the base symbolic function will be invoked.

    For example:

    ```
    @quantized_args(True, False)
    def foo(g, x, y):
        return x + y
    ```

    is equivalent to

    ```
    def q_foo(g, x, y):
        if is_quantized_tensor(x):
            x = dequantize(x)
            out = foo(g, x, y)
            return quantize(out)
        else:
            return foo(g, x, y)
    ```

    Args:
        arg_q_descriptors: A sequence of bool, where each element represents if the
          argument is QTensor for quantized version of this operator. It defaults
          to False for unspecified (variable length) arguments.
        scale: Quantized output scale. If None, derive from
          the first quantized input scale.
        zero_point: Quantized output zero point. If None,
          derive from the first quantized input zero point.
        quantize_output: If True, quantize the output of the base operator. Default is True
    """

    def decorator(fn):

        @functools.wraps(fn)
        def wrapper(g, *args, **kwargs):
            nonlocal scale
            nonlocal zero_point
            if scale is not None:
                _scale = g.op('Constant', value_t=torch.tensor(scale))
            else:
                _scale = None
            if zero_point is not None:
                _zero_point = g.op('Constant', value_t=torch.tensor(zero_point))
            else:
                _zero_point = None
            arg_q_descriptors_extended = arg_q_descriptors + (False,) * (len(args) - len(arg_q_descriptors))
            descriptor_args = tuple(zip(arg_q_descriptors_extended, args))

            def _is_arg_quantized(descriptor, arg):
                return descriptor and _is_value(arg) and _is_tuple_construct(arg)
            is_quantized = list()
            for descriptor, arg in descriptor_args:
                if _is_packed_list(arg):
                    for arg_input in arg.node().inputs():
                        is_quantized.append(_is_arg_quantized(descriptor, arg_input))
                else:
                    is_quantized.append(_is_arg_quantized(descriptor, arg))
            if not any(is_quantized):
                return fn(g, *args, **kwargs)
            non_quantized_args = []
            for descriptor, arg in descriptor_args:
                if _is_arg_quantized(descriptor, arg):
                    dequantized_arg, arg_scale, arg_zero_point, _ = dequantize_helper(g, arg)
                    non_quantized_args.append(dequantized_arg)
                    if _scale is None:
                        _scale = arg_scale
                    if _zero_point is None:
                        _zero_point = arg_zero_point
                elif _is_packed_list(arg):
                    for arg_input in arg.node().inputs():
                        if _is_arg_quantized(descriptor, arg_input):
                            dequantized_arg, arg_scale, arg_zero_point, _ = dequantize_helper(g, arg_input)
                            if _scale is None:
                                _scale = arg_scale
                            if _zero_point is None:
                                _zero_point = arg_zero_point
                            arg_input.replaceAllUsesWith(dequantized_arg)
                    non_quantized_args.append(arg)
                else:
                    non_quantized_args.append(arg)
            output = fn(g, *non_quantized_args, **kwargs)
            assert _scale is not None, 'Bug: Scale must be set for quantized operator'
            assert _zero_point is not None, 'Bug: Zero point must be set for quantized operator'
            if quantize_output:
                return quantize_helper(g, output, _scale, _zero_point)
            return output
        return wrapper
    return decorator