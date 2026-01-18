import collections
import logging
import operator
from typing import Any, DefaultDict, Deque, Dict, Iterator, List, Optional, Set, Tuple
import torch
from torch._dynamo.utils import counters
from torch._utils_internal import print_graph
from .. import config
from ..pattern_matcher import (
def decompose_stack(graph: torch.fx.GraphModule, input_tensors: List[Any]) -> Any:
    unsqueezed_inputs = []
    for input_tensor in input_tensors:
        unsqueezed_input = graph.call_function(aten.unsqueeze, args=(input_tensor, 0))
        unsqueezed_inputs.append(unsqueezed_input)
    stacked_inputs = graph.call_function(aten.cat, args=(unsqueezed_inputs, 0))
    return stacked_inputs