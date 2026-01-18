import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.nn.quantized.dynamic as nnqd
import torch.ao.nn.quantized as nnq
import torch.ao.nn.intrinsic.qat as nniqat
import torch.ao.nn.qat as nnqat
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.intrinsic.quantized as nniq
from torch.fx import GraphModule
from torch.fx.graph import Node
from .utils import (
from .ns_types import (
from typing import List, Optional, Dict, Callable
def get_qconv_fun_weight(node: Node, gm: GraphModule) -> torch.Tensor:
    qconv_state_node = node.args[1]
    assert isinstance(qconv_state_node, Node)
    assert qconv_state_node.op == 'get_attr'
    qconv_state_obj = getattr_from_fqn(gm, qconv_state_node.target)
    return qconv_state_obj.weight()