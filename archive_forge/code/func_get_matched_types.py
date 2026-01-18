import torch
from torch.ao.quantization.backend_config import BackendConfig
from torch.fx.graph import Node, Graph
from ..utils import _parent_name, NodePattern, Pattern
from ..fuser_method_mappings import get_fuser_method_new
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Union
from .custom_config import FuseCustomConfig
from .match_utils import MatchAllNode
from torch.nn.utils.parametrize import type_before_parametrizations
def get_matched_types(m):
    if isinstance(m, tuple):
        return tuple(map(get_matched_types, m))
    if isinstance(m, torch.nn.Module):
        return type_before_parametrizations(m)
    return m