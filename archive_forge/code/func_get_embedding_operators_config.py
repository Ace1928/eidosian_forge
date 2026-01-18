from __future__ import annotations
import copy
from typing import List, Set
import torch
import torch.nn.functional as F
from torch.ao.quantization.observer import PerChannelMinMaxObserver
from torch.ao.quantization.quantizer.quantizer import (
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import (
def get_embedding_operators_config() -> OperatorConfig:
    weight_quantization_spec = QuantizationSpec(dtype=torch.uint8, qscheme=torch.per_channel_affine_float_qparams, ch_axis=0, observer_or_fake_quant_ctr=PerChannelMinMaxObserver.with_args(eps=2 ** (-12)))
    quantization_config = QuantizationConfig(None, None, weight_quantization_spec, None)
    ops: List[OperatorPatternType] = [[torch.nn.Embedding]]
    ops.append([F.embedding])
    supported_config_and_operators = OperatorConfig(config=quantization_config, operators=ops)
    return copy.deepcopy(supported_config_and_operators)