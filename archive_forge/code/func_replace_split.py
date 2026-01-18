import itertools
import logging
import operator
from typing import Any, Callable, List, Optional, Sequence, Set, Tuple, Union
from typing_extensions import TypeAlias
import torch
from torch._dynamo.utils import counters
from ..pattern_matcher import (
from .pre_grad import (
def replace_split(self, graph: torch.fx.Graph, split_node: torch.fx.Node, split_sections: List[int], user_inputs_list: List[List[Union[torch.fx.Node, _Range]]], split_ranges: List[_Range]) -> List[List[torch.fx.Node]]:
    """
        Replace the split node. It can either remove the split node if len(split_ranges) == 1, or simplify it
        into a split with lesser sections if len(split_ranges) > 1.

        Returns the new `user_inputs_list`, with tuples replaced with new getitems from the newer split node.
        """
    split_input = split_node.args[0]
    split_dim = split_node.kwargs['dim']
    if len(split_ranges) == 1:
        split_items = [split_input]
    else:
        with graph.inserting_after(split_node):
            new_split = graph.call_function(torch.split, args=(split_input, [r[1] - r[0] for r in split_ranges]), kwargs={'dim': split_dim})
            new_split.meta.update(split_node.meta)
            counters['inductor']['scmerge_split_added'] += 1
        with graph.inserting_after(new_split):
            split_items = [graph.call_function(operator.getitem, args=(new_split, i)) for i in range(len(split_ranges))]
    cumulative_sizes = [0] + torch.cumsum(torch.tensor(split_sections), 0).tolist()
    new_user_inputs_list = []
    for user_inputs in user_inputs_list:
        new_user_inputs = []
        for user_input in user_inputs:
            if isinstance(user_input, tuple):
                new_user_inputs.append(split_items[split_ranges.index((cumulative_sizes[user_input[0]], cumulative_sizes[user_input[1] + 1]))])
            else:
                new_user_inputs.append(user_input)
        new_user_inputs_list.append(new_user_inputs)
    return new_user_inputs_list