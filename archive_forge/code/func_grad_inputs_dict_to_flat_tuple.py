import torch
import torch.utils._pytree as pytree
from collections import namedtuple
import functools
def grad_inputs_dict_to_flat_tuple(grad_inputs_dict, args_info):
    result = []
    for name, arg_info in args_info._asdict().items():
        if name not in grad_inputs_dict:
            result.append(pytree.tree_map(lambda x: None, arg_info))
            continue
        result.append(grad_inputs_dict[name])
    return tuple(pytree.tree_leaves(result))