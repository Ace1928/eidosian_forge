import functools
import torch
import torch._C._onnx as _C_onnx
from torch.onnx import (
from torch.onnx._internal import _beartype, jit_utils, registration
@_onnx_symbolic('aten::fake_quantize_per_channel_affine')
@symbolic_helper.parse_args('v', 'v', 'v', 'i', 'i', 'i')
@_beartype.beartype
def fake_quantize_per_channel_affine(g: jit_utils.GraphContext, inputs, scale, zero_point, axis, quant_min=-128, quant_max=127):
    if (quant_min, quant_max) not in [(0, 255), (-128, 127), (0, 127)]:
        raise errors.SymbolicValueError(f'For (quant_min, quant_max), ONNX allows only (0, 127), (0, 255) and (-128, 127). Got ({quant_min}, {quant_max})', inputs)
    if quant_min == 0:
        zero_point = g.op('Cast', zero_point, to_i=_C_onnx.TensorProtoDataType.UINT8)
    else:
        zero_point = g.op('Cast', zero_point, to_i=_C_onnx.TensorProtoDataType.INT8)
    quantized = g.op('QuantizeLinear', inputs, scale, zero_point, axis_i=axis)
    if (quant_min, quant_max) == (0, 127):
        quantized = g.op('Clip', quantized, opset9.unused(g), g.op('Constant', value_t=torch.tensor(127, dtype=torch.uint8)))
    return g.op('DequantizeLinear', quantized, scale, zero_point, axis_i=axis)