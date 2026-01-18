import torch
from torch.fx import map_arg, Node
from torch.fx.graph import Graph
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.intrinsic.quantized as nniq
import torch.ao.nn.intrinsic.quantized.dynamic as nniqd
import torch.ao.nn.quantized as nnq
import torch.ao.nn.quantized.dynamic as nnqd
import torch.ao.nn.quantized.reference as nnqr
from torch.ao.nn.quantized.modules.utils import WeightedQuantizedModule
from torch.fx import GraphModule
from .utils import (
from ..utils import _parent_name
from ..qconfig import QConfigAny
from ..quantization_mappings import get_quantized_operator
from .utils import create_node_from_old_node_preserve_meta
from typing import Dict, Tuple, Type, List, Callable, Any, Union, Set, Optional
import operator
def is_default_node(node, modules):
    func_list = [torch.nn.functional.elu, torch.nn.functional.hardswish, torch.nn.functional.instance_norm, torch.nn.functional.layer_norm, torch.nn.functional.leaky_relu, torch.nn.functional.dropout]
    method_list: List[Any] = []
    module_type_list = [nnqr.ConvTranspose1d, nnqr.ConvTranspose2d, nnqr.ConvTranspose3d, torch.nn.ELU, torch.nn.LeakyReLU, torch.nn.Hardswish, torch.nn.InstanceNorm1d, torch.nn.InstanceNorm2d, torch.nn.InstanceNorm3d, torch.nn.LayerNorm, torch.nn.Dropout, torch.nn.PReLU, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d, torch.ao.nn.intrinsic.BNReLU2d, torch.ao.nn.intrinsic.BNReLU3d]
    return _is_node_in_list(node, modules, func_list, method_list, module_type_list)