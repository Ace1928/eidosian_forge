import copy
import torch
import torch.nn as nn
from torch.ao.quantization import (
from torch.ao.quantization.backend_config import (
from torch.ao.quantization.fake_quantize import (
from torch.ao.quantization.observer import (
from torch.ao.quantization.qconfig import (
from torch.ao.quantization.stubs import DeQuantStub
from torch.ao.quantization.utils import (
from torch.ao.quantization.observer import _is_activation_post_process
from torch.ao.quantization.qconfig_mapping import QConfigMapping
from torch.fx import GraphModule, map_arg
from torch.fx.graph import (
from .custom_config import PrepareCustomConfig
from ._decomposed import quantized_decomposed_lib  # noqa: F401
from typing import Callable, Optional, List, Dict, Any, Set, Tuple, Union, Type
from dataclasses import dataclass
from collections import namedtuple
import operator
import warnings
def get_non_observable_arg_indexes_and_types(node: Node) -> Dict[Union[type, torch.dtype], Callable[[Node], List[int]]]:
    """
    Returns a dict with of non float tensor types as keys and values which correspond to a
    function to retrieve the list (which takes the node as an argument)
    """
    info = NodeInfo(node.op, node.target)
    return NON_OBSERVABLE_ARG_DICT.get(info, EMPTY_ARG_DICT)