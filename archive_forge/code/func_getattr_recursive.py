import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, Union
import torch.fx
from torch.fx._compatibility import compatibility
from torch.fx.graph import map_arg
from torch.fx.passes.utils import HolderModule, lift_subgraph_as_module
from .tools_common import NodeList
@compatibility(is_backward_compatible=False)
def getattr_recursive(obj, name):
    for layer in name.split('.'):
        if hasattr(obj, layer):
            obj = getattr(obj, layer)
        else:
            return None
    return obj