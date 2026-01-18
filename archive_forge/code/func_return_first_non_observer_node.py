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
def return_first_non_observer_node(node: Node, gm: GraphModule) -> Node:
    """
    If node is not an observer, returns it.  If node is an observer,
    navigates up the graph and returns the first parent which is not an
    observer.  For example,

    graph: (node_non_obs), node = node_non_obs : returns node_non_obs
    graph: (node_non_obs -> obs0), node = obs0 : returns node_non_obs
    graph: (node_non_obs -> obs0 -> fq0), node = fq0 : returns node_non_obs
    """
    if node.op == 'call_module':
        node_obj = getattr_from_fqn(gm, node.target)
        if _is_activation_post_process(node_obj):
            assert len(node.args) == 1
            assert isinstance(node.args[0], Node)
            node = node.args[0]
            assert isinstance(node.target, str)
            node_obj = getattr_from_fqn(gm, node.target)
            if _is_activation_post_process(node_obj):
                assert len(node.args) == 1
                assert isinstance(node.args[0], Node)
                node = node.args[0]
    return node