import copy
from typing import (
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils._named_member_accessor import NamedMemberAccessor
def create_names_map(named_params: Union[Dict[str, Tensor], Iterable[Tuple[str, Tensor]]], tied_named_params: Union[Dict[str, Tensor], Iterable[Tuple[str, Tensor]]]) -> Dict[str, List[str]]:
    """
    named_params is a dictionary of tensors: {'A': A, 'B': B}
    tied_named_params is another dictionary of tensors {'A': A, 'B': B, 'B_tied': B}
    with potentially tied (or 'duplicated') tensors

    This function creates a mapping from the names in named_params to the
    names in tied_named_params: {'A': ['A'], 'B': ['B', 'B_tied']}.
    """
    named_params = dict(named_params)
    tied_named_params = dict(tied_named_params)
    tensors_dict_keys = set(named_params.keys())
    tied_tensors_dict_keys = set(tied_named_params.keys())
    assert tensors_dict_keys.issubset(tied_tensors_dict_keys)
    tensor_to_mapping: Dict[Tensor, Tuple[str, List[str]]] = {}
    for key, tensor in named_params.items():
        tensor_to_mapping[tensor] = (key, [])
    for key, tensor in tied_named_params.items():
        assert tensor in tensor_to_mapping
        tensor_to_mapping[tensor][1].append(key)
    return dict(tensor_to_mapping.values())