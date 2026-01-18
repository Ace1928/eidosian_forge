import itertools
import logging
import operator
from typing import Any, Callable, List, Optional, Sequence, Set, Tuple, Union
from typing_extensions import TypeAlias
import torch
from torch._dynamo.utils import counters
from ..pattern_matcher import (
from .pre_grad import (
def get_merged_user_inputs(self, split_node: torch.fx.Node, cat_node: torch.fx.Node) -> List[Union[torch.fx.Node, _Range]]:
    user_inputs = get_arg_value(cat_node, 0, 'tensors')
    simplified_user_inputs = []
    split_users = set(split_node.users.keys())
    for user_input in user_inputs:
        if user_input not in split_users:
            simplified_user_inputs.append(user_input)
        else:
            simplified_user_inputs.append(user_input.args[1])
    return self.merge_consecutive_inputs(simplified_user_inputs)