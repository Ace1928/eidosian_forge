import copy
import functools
import math
import operator
from typing import Any, Tuple
import torch
from torch._dynamo.utils import counters
from torch.fx.experimental.symbolic_shapes import has_free_symbols
from ..lowering import lowerings as L, require_channels_last
from ..pattern_matcher import Arg, CallFunction, filter_nodes, KeywordArg, ListOf, Match
from ..utils import pad_listlike
from .freezing_patterns import register_freezing_graph_pattern
from .post_grad import register_lowering_pattern
def generate_pattern_with_output_quant(computation_call, dtype=torch.float32):
    """
    quantize output:
        output = round(output * o_inv_scale)
        output = output + zero_point
        output = clamp_min(output, 0)
        output = clamp_max(output, 127)
        output = output.to(uint8)
    """
    assert dtype in [torch.float32, torch.bfloat16]
    quantized_op_output_pattern_pt2e = CallFunction(prims.convert_element_type.default, CallFunction(aten.clamp_max.default, CallFunction(aten.clamp_min.default, CallFunction(aten.add.Tensor, CallFunction(aten.round.default, CallFunction(aten.mul.Tensor, _may_generate_pattern_with_dtype_convert(computation_call, KeywordArg('autocast_output_quant_dtype'), dtype != torch.float32), KeywordArg('o_inv_scale'))), KeywordArg('o_zp')), KeywordArg('o_qmin')), KeywordArg('o_qmax')), KeywordArg('o_dtype'))
    return quantized_op_output_pattern_pt2e