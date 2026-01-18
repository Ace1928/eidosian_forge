import itertools
import operator
from dataclasses import dataclass
from typing import Callable, Dict, List, NamedTuple, Optional
import torch
import torch.nn.functional as F
from torch.ao.quantization.fx.utils import get_new_attr_name_with_prefix
from torch.ao.quantization.pt2e.graph_utils import find_sequential_partitions
from torch.ao.quantization.pt2e.utils import (
from torch.ao.quantization.quantizer import (
from torch.ao.quantization.quantizer.utils import (
from torch.fx import Node
from torch.fx.passes.utils.matcher_with_name_node_map_utils import (
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions
def get_output_act_qspec(quantization_config: Optional[QuantizationConfig]):
    if quantization_config is None:
        return None
    if quantization_config.output_activation is None:
        return None
    quantization_spec: QuantizationSpec = quantization_config.output_activation
    assert quantization_spec.qscheme in [torch.per_tensor_affine, torch.per_tensor_symmetric]
    return quantization_spec