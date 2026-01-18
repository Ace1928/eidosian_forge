from abc import ABC
from typing import Callable, Dict, List, Optional, Type
import torch
from torch.ao.quantization.backend_config import (
from torch.ao.quantization.utils import NodePattern, Pattern, QuantizerCls
from torch.fx.graph import Node
from .utils import all_node_args_have_no_tensors
def is_general_tensor_value_op(self) -> bool:
    return self.observation_type == ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT