import itertools
import logging
import operator
from typing import Any, Callable, List, Optional, Sequence, Set, Tuple, Union
from typing_extensions import TypeAlias
import torch
from torch._dynamo.utils import counters
from ..pattern_matcher import (
from .pre_grad import (
@register_graph_pattern(CallFunction([torch.stack, torch.cat], tensors=getitem_split, dim=Ignored(), _users=MULTIPLE), pass_dict=split_cat_pass, extra_check=config_flag('split_cat_fx_passes'))
@register_graph_pattern(CallFunction([torch.stack, torch.cat], getitem_split, dim=Ignored(), _users=MULTIPLE), pass_dict=split_cat_pass, extra_check=config_flag('split_cat_fx_passes'))
@register_graph_pattern(CallFunction([torch.stack, torch.cat], getitem_split, Ignored(), _users=MULTIPLE), pass_dict=split_cat_pass, extra_check=config_flag('split_cat_fx_passes'))
def simplify_split_cat(match: Match, split_sections: List[int], dim: int):
    if not isinstance(split_sections, (list, tuple)):
        return
    split_node = next((node for node in match.nodes if node.target == torch.split))
    SplitCatSimplifier().simplify(match.graph, split_node, split_sections)