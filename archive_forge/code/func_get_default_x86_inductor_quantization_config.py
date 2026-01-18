import copy
import functools
import itertools
import operator
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import torch
import torch.nn.functional as F
from torch.ao.quantization.fake_quantize import FusedMovingAvgObsFakeQuantize
from torch.ao.quantization.observer import (
from torch.ao.quantization.pt2e.graph_utils import find_sequential_partitions
from torch.ao.quantization.qconfig import _ObserverOrFakeQuantizeConstructor
from torch.ao.quantization.quantizer.quantizer import (
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import (
from torch.fx import Node
from torch.fx.passes.utils.source_matcher_utils import (
@functools.lru_cache
def get_default_x86_inductor_quantization_config(is_qat: bool=False):
    act_observer_or_fake_quant_ctr: _ObserverOrFakeQuantizeConstructor = FusedMovingAvgObsFakeQuantize if is_qat else HistogramObserver
    act_quantization_spec = QuantizationSpec(dtype=torch.uint8, quant_min=0, quant_max=255, qscheme=torch.per_tensor_affine, is_dynamic=False, observer_or_fake_quant_ctr=act_observer_or_fake_quant_ctr.with_args(eps=2 ** (-12)))
    weight_observer_or_fake_quant_ctr: _ObserverOrFakeQuantizeConstructor = FusedMovingAvgObsFakeQuantize if is_qat else PerChannelMinMaxObserver
    extra_args: Dict[str, Any] = {'eps': 2 ** (-12)}
    if is_qat:
        extra_args['observer'] = MovingAveragePerChannelMinMaxObserver
    weight_quantization_spec = QuantizationSpec(dtype=torch.int8, quant_min=-128, quant_max=127, qscheme=torch.per_channel_symmetric, ch_axis=0, is_dynamic=False, observer_or_fake_quant_ctr=weight_observer_or_fake_quant_ctr.with_args(**extra_args))
    bias_observer_or_fake_quant_ctr: _ObserverOrFakeQuantizeConstructor = PlaceholderObserver
    bias_quantization_spec = QuantizationSpec(dtype=torch.float, observer_or_fake_quant_ctr=bias_observer_or_fake_quant_ctr)
    quantization_config = QuantizationConfig(act_quantization_spec, act_quantization_spec, weight_quantization_spec, bias_quantization_spec, is_qat)
    return quantization_config