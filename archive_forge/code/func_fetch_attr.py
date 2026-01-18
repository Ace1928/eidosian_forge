import torch
import torch.fx
from torch.fx import (
from torch.ao.ns.fx.utils import (
from torch.ao.ns.fx.ns_types import (
from torch.ao.ns.fx.graph_passes import _maybe_get_fqn
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.qconfig import QConfigAny
from torch.ao.quantization.utils import getattr_from_fqn
from torch.ao.quantization.fx.match_utils import _MatchResult
from torch.utils._pytree import tree_map
import collections
import copy
from typing import List, Dict, Set, Tuple, Callable, Any, Optional
import operator
def fetch_attr(target: str):
    target_atoms = target.split('.')
    attr_itr = self.mod
    for i, atom in enumerate(target_atoms):
        if not hasattr(attr_itr, atom):
            raise RuntimeError(f'Node referenced nonexistent target {'.'.join(target_atoms[:i])}')
        attr_itr = getattr(attr_itr, atom)
    return attr_itr