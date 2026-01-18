import itertools
import logging
import operator
from typing import Any, Callable, List, Optional, Sequence, Set, Tuple, Union
from typing_extensions import TypeAlias
import torch
from torch._dynamo.utils import counters
from ..pattern_matcher import (
from .pre_grad import (
@register_graph_pattern(CallFunctionVarArgs(torch.split, users=MULTIPLE), pass_dict=normalization_pass, extra_check=config_flag('split_cat_fx_passes'))
@register_graph_pattern(CallMethodVarArgs('split', users=MULTIPLE), pass_dict=normalization_pass, extra_check=config_flag('split_cat_fx_passes'))
def normalize_split_default(match: Match, *args, **kwargs):
    return normalize_split_base(match, _get_split_args_default)