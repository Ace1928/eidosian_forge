from itertools import chain
from operator import getitem
import torch
import torch.nn.functional as F
from torch import nn
from torch.fx import symbolic_trace
from torch.nn.utils import parametrize
from typing import Type, Set, Dict, Callable, Tuple, Optional, Union
from torch.ao.pruning import BaseSparsifier
from .parametrization import FakeStructuredSparsity, BiasHook, module_contains_param
from .match_utils import apply_match, MatchAllNode
from .prune_functions import (
def make_config_from_model(self, model: nn.Module, SUPPORTED_MODULES: Optional[Set[Type]]=None) -> None:
    if SUPPORTED_MODULES is None:
        SUPPORTED_MODULES = _get_supported_structured_pruning_modules()
    super().make_config_from_model(model, SUPPORTED_MODULES=SUPPORTED_MODULES)