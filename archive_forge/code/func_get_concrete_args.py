import builtins
import collections
import functools
import inspect
import math
import operator
import os
import random
import warnings
from typing import Any, Callable, Dict, List, Optional, Type, Union
import torch
from torch import nn
from torch.fx import Graph, GraphModule, Proxy, Tracer
from torch.fx._compatibility import compatibility
from torch.fx.proxy import ParameterProxy
from .. import PretrainedConfig, PreTrainedModel, logging
from ..models.auto import get_values
from ..models.auto.modeling_auto import (
from ..pytorch_utils import is_torch_greater_or_equal_than_2_0
from ..utils import (
def get_concrete_args(model: nn.Module, input_names: List[str]):
    sig = inspect.signature(model.forward)
    if not set(input_names) <= set(sig.parameters.keys()):
        formatted_input_names = input_names[0] if len(input_names) == 1 else ', '.join(input_names)
        formatted_allowed_input_names = ', '.join(sig.parameters.keys())
        raise ValueError(f'The model does not have input(s) named: {formatted_input_names}, expected a subset of the following: {formatted_allowed_input_names}')
    return {p.name: p.default for p in sig.parameters.values() if p.name not in input_names}