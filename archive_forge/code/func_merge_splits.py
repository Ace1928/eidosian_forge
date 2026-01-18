import itertools
import logging
import operator
from typing import Any, Callable, List, Optional, Sequence, Set, Tuple, Union
from typing_extensions import TypeAlias
import torch
from torch._dynamo.utils import counters
from ..pattern_matcher import (
from .pre_grad import (
@register_graph_pattern(TorchSplit(CallFunction(operator.getitem, TorchSplit(KeywordArg('first_split_input'), KeywordArg('first_split_sections')), Ignored()), KeywordArg('next_split_sections')), pass_dict=merge_splits_pass, extra_check=config_flag('split_cat_fx_passes'))
def merge_splits(match: Match, first_split_input: torch.fx.Node, first_split_sections: List[int], next_split_sections: List[int], dim: int):
    node = match.output_node()
    graph = match.graph
    first_split = node.args[0].args[0]
    next_split_index = node.args[0].args[1]
    new_split_sections = list(first_split_sections)
    new_split_sections[next_split_index:next_split_index + 1] = next_split_sections
    first_split_dim = first_split.kwargs['dim']
    to_remove = []
    with graph.inserting_before(first_split):
        new_split = graph.call_function(torch.split, args=(first_split_input, new_split_sections), kwargs={'dim': first_split_dim})
        first_split_num_to_user = {user.args[1]: user for user in first_split.users.keys()}
        new_split_num = 0
        for split_num in range(len(first_split_sections)):
            if split_num not in first_split_num_to_user:
                new_split_num += 1
                continue
            old_getitem = first_split_num_to_user[split_num]
            if split_num != next_split_index:
                old_getitem.update_arg(0, new_split)
                old_getitem.update_arg(1, new_split_num)
                new_split_num += 1
            else:
                next_split_num_to_user = {user.args[1]: user for user in node.users.keys()}
                for next_split_num in range(len(next_split_sections)):
                    with graph.inserting_after(new_split):
                        new_getitem = graph.call_function(operator.getitem, args=(new_split, new_split_num))
                    new_split_num += 1
                    next_getitem = next_split_num_to_user[next_split_num]
                    new_getitem.meta.update(next_getitem.meta)
                    next_getitem.replace_all_uses_with(new_getitem)
                    to_remove.append(next_getitem)
                to_remove.append(node)
                to_remove.append(old_getitem)
        to_remove.append(first_split)
    for node in to_remove:
        graph.erase_node(node)
    counters['inductor']['consecutive_split_merged'] += 1