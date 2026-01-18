import enum
import operator
import torch
import torch.nn as nn
import torch.ao.nn.intrinsic.quantized as nniq
import torch.ao.nn.quantized as nnq
from typing import Tuple, Callable, Dict, Set, List, Optional, Union
from torch.fx import GraphModule
from torch.fx.graph import Node
from torch.ao.quantization import (
from torch.ao.quantization.utils import getattr_from_fqn
from torch.ao.quantization.observer import _is_activation_post_process
from .ns_types import NSNodeTargetType, NSResultsType
def get_normalized_nth_input(node: Node, gm: GraphModule, idx: int) -> Node:
    """
    Given a node, gets the n'th input to that node, normalizing
    args and kwargs to the best of its ability.
    """
    try:
        norm_args_and_kwargs = node.normalized_arguments(gm, normalize_to_only_use_kwargs=True)
        if norm_args_and_kwargs is not None:
            norm_args, norm_kwargs = norm_args_and_kwargs
            assert len(norm_args) + len(norm_kwargs) > idx
            if idx < len(norm_args):
                return norm_args[idx]
            else:
                return list(norm_kwargs.values())[idx]
        else:
            assert len(node.args) + len(node.kwargs) > idx
            if idx < len(node.args):
                return node.args[idx]
            else:
                kwargs_idx = idx + len(node.args)
                return list(node.kwargs.values())[kwargs_idx]
    except RuntimeError:
        assert len(node.args) + len(node.kwargs) > idx
        if idx < len(node.args):
            return node.args[idx]
        else:
            kwargs_idx = idx + len(node.args)
            return list(node.kwargs.values())[kwargs_idx]