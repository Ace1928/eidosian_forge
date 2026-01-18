import itertools
import logging
import operator
from typing import Any, Callable, List, Optional, Sequence, Set, Tuple, Union
from typing_extensions import TypeAlias
import torch
from torch._dynamo.utils import counters
from ..pattern_matcher import (
from .pre_grad import (
@register_graph_pattern(CallFunction(torch.tanh, CallFunction(torch.stack, getitem_split, dim=Ignored(), _users=1), _users=1), pass_dict=merge_getitem_cat_pass, extra_check=config_flag('split_cat_fx_passes'))
@register_graph_pattern(CallFunction(torch.tanh, CallFunction(torch.stack, tensors=getitem_split, dim=Ignored(), _users=1), _users=1), pass_dict=merge_getitem_cat_pass, extra_check=config_flag('split_cat_fx_passes'))
@register_graph_pattern(CallFunction(torch.tanh, CallFunction(torch.stack, getitem_split, Ignored(), _users=1), _users=1), pass_dict=merge_getitem_cat_pass, extra_check=config_flag('split_cat_fx_passes'))
def merge_stack_tahn_unbind(match: Match, split_sections: List[int], dim: int):
    if not isinstance(split_sections, (list, tuple)):
        return
    graph = match.graph
    split_node = next((node for node in match.nodes if node.target == torch.split))
    split_input, split_size, split_dim = _get_split_args_default(split_node)
    next_users = find_next_users(split_node)
    split_sections = list(split_sections)
    for user in next_users:
        if user.target == torch.stack:
            if not safe_to_abort_node(user):
                continue
            unbind_user = find_next_users(user)[0]
            if unbind_user.target != torch.unbind:
                continue
            unbind_dim = get_arg_value(unbind_user, 1, 'dim') or 0
            stack_dim = get_arg_value(user, 1, 'dim') or 0
            if unbind_user.target != torch.unbind or stack_dim != unbind_dim:
                continue
            indices = []
            split_sections_for_unbind = []
            for arg in user.args[0]:
                indices.append(arg.args[1])
                split_sections_for_unbind.append(split_sections[arg.args[1]])
            indices.sort()
            if indices[len(indices) - 1] - indices[0] + 1 != len(indices):
                continue
            user.update_arg(0, user.args[0][0])
            fused_tensor_size = 0
            for i in range(len(split_node.args[1])):
                if i in indices:
                    fused_tensor_size += split_node.args[1][i]
            split_sections[indices[0]] = fused_tensor_size
            for i in indices[1:]:
                split_sections[i] = 0
            new_split_sections, index_mapping = remove_zeros(split_sections)
            with graph.inserting_after(split_node):
                new_split_node = graph.call_function(torch.split, args=(split_input, split_sections), kwargs={'dim': split_dim})
                replace_unbind_with_split = graph.call_function(torch.split, args=(unbind_user.args[0], split_sections_for_unbind), kwargs={'dim': split_dim})
                unbind_user.replace_all_uses_with(replace_unbind_with_split)
                replace_unbind_with_split.meta.update(unbind_user.meta)
                split_node.replace_all_uses_with(new_split_node)
                new_split_node.meta.update(split_node.meta)
                to_remove = [unbind_user]
                new_split_getitem_nodes = list(new_split_node.users.keys())
                for getitem_node in new_split_getitem_nodes:
                    if getitem_node.args[1] in indices[1:]:
                        to_remove.append(getitem_node)
                    elif getitem_node.args[1] == indices[0]:
                        user.replace_all_uses_with(getitem_node)
                        getitem_node.meta.update(user.meta)
                    else:
                        getitem_node.update_arg(1, index_mapping[getitem_node.args[1]])
                graph.erase_node(split_node)
                graph.erase_node(user)
                for getitem_node in to_remove:
                    graph.erase_node(getitem_node)
                new_split_node.update_arg(1, new_split_sections)
                split_node = new_split_node
                split_sections = new_split_sections
                counters['inductor']['stack_tahn_unbind_merged'] += 1