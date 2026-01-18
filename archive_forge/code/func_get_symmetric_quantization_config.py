from __future__ import annotations
import copy
import functools
from typing import Any, Callable, Dict, List, Optional, Set
import torch
import torch._dynamo as torchdynamo
import torch.nn.functional as F
from torch.ao.quantization.fake_quantize import (
from torch.ao.quantization.observer import (
from torch.ao.quantization.qconfig import _ObserverOrFakeQuantizeConstructor
from torch.ao.quantization.quantizer import QuantizationSpec, Quantizer
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import (
from torch.fx import Node
@functools.lru_cache
def get_symmetric_quantization_config(is_per_channel: bool=False, is_qat: bool=False, is_dynamic: bool=False):
    extra_args: Dict[str, Any] = {'eps': 2 ** (-12)}
    if is_qat:
        if is_dynamic:
            act_observer_or_fake_quant_ctr = FakeQuantize
            dynamic_quant_observer = MovingAverageMinMaxObserver.with_args(averaging_constant=1)
            extra_args['observer'] = dynamic_quant_observer
        else:
            act_observer_or_fake_quant_ctr = FusedMovingAvgObsFakeQuantize
    elif is_dynamic:
        act_observer_or_fake_quant_ctr = PlaceholderObserver
    else:
        act_observer_or_fake_quant_ctr = HistogramObserver
    act_quantization_spec = QuantizationSpec(dtype=torch.int8, quant_min=-128, quant_max=127, qscheme=torch.per_tensor_affine, is_dynamic=is_dynamic, observer_or_fake_quant_ctr=act_observer_or_fake_quant_ctr.with_args(**extra_args))
    weight_qscheme = torch.per_channel_symmetric if is_per_channel else torch.per_tensor_symmetric
    weight_observer_or_fake_quant_ctr: _ObserverOrFakeQuantizeConstructor = MinMaxObserver
    if is_qat:
        weight_observer_or_fake_quant_ctr = FusedMovingAvgObsFakeQuantize
    elif is_per_channel:
        weight_observer_or_fake_quant_ctr = PerChannelMinMaxObserver
    extra_args: Dict[str, Any] = {'eps': 2 ** (-12)}
    if is_qat:
        if weight_qscheme == torch.per_tensor_symmetric:
            extra_args['observer'] = MovingAverageMinMaxObserver
        else:
            extra_args['observer'] = MovingAveragePerChannelMinMaxObserver
    weight_quantization_spec = QuantizationSpec(dtype=torch.int8, quant_min=-127, quant_max=127, qscheme=weight_qscheme, ch_axis=0, is_dynamic=False, observer_or_fake_quant_ctr=weight_observer_or_fake_quant_ctr.with_args(**extra_args))
    bias_quantization_spec = None
    if is_dynamic:
        quantization_config = QuantizationConfig(act_quantization_spec, None, weight_quantization_spec, bias_quantization_spec, is_qat)
    else:
        quantization_config = QuantizationConfig(act_quantization_spec, act_quantization_spec, weight_quantization_spec, bias_quantization_spec, is_qat)
    return quantization_config