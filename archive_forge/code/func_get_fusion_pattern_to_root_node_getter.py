from typing import Dict, Any, List, Callable, Union, Tuple, Type
import torch
import torch.nn as nn
import torch.nn.functional as F
from .backend_config import (
from ..utils import Pattern
from ..fuser_method_mappings import (
def get_fusion_pattern_to_root_node_getter(backend_config: BackendConfig) -> Dict[Pattern, Callable]:
    """ Get a map from fusion pattern to a function that returns the root node
    from the fusion pattern, e.g. the most common one is:
    def get_root_node(node_pattern):
        while not isinstance(node_pattern[-1], Node):
            node_pattern = node_pattern[-1]
        return node_pattern[-1]
    This can work for all patterns whose root node is the "last node" in the pattern,
    e.g. (torch.add, MatchAllNode, (torch.ReLU, torch.Conv2d))
    """
    root_node_getter_mapping: Dict[Pattern, Callable] = {}
    for pattern, config in backend_config._pattern_complex_format_to_config.items():
        if config._root_node_getter is not None:
            root_node_getter_mapping[pattern] = config._root_node_getter
    return root_node_getter_mapping