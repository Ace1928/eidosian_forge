import itertools
import logging
import operator
from typing import Any, Callable, List, Optional, Sequence, Set, Tuple, Union
from typing_extensions import TypeAlias
import torch
from torch._dynamo.utils import counters
from ..pattern_matcher import (
from .pre_grad import (
def get_user_input_list(self, split_node: torch.fx.Node, next_users: List[torch.fx.Node]) -> List[List[Union[torch.fx.Node, _Range]]]:
    """
        Returns list of inputs to the following user nodes, in order. The outer list represents the user node. The inner
        list represents the inputs to that particular node. This list can either contain
          - a tuple representing the ranges of get_items that should go into the cat (closed interval)
          - torch.fx.Node representing "other" inputs (which are not coming from our split)
        """
    user_inputs_list: List[List[Union[torch.fx.Node, _Range]]] = []
    for user in next_users:
        if user.target in {torch.cat, torch.stack}:
            user_inputs_list.append(self.get_merged_user_inputs(split_node, user))
        else:
            user_inputs_list.append(self.get_non_cat_node_input(split_node, user))
    return user_inputs_list