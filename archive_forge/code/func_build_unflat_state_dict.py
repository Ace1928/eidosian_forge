import copy
from itertools import groupby
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Tuple, cast
import torch
from fairscale.nn.misc import FlattenParamsWrapper
def build_unflat_state_dict(instance_list: List['FullyShardedDataParallel'], world_pad_info: List[List[List[int]]], state: Dict[int, Dict[str, List[torch.Tensor]]], singleton_state: Dict[int, Dict[str, List[torch.Tensor]]], uncollected_opt_state: Dict[int, Dict], original_sd: Dict) -> Dict:
    """Build an unflattened optimizer state dict given a list of flattened optimizer state dicts
    from each rank. This is only called on rank 0.

    Args:
        instance_list: list of FSDP wrapper objects
        world_pad_info: [param_id][fsdp_instance_id][bytes_padded_per_rank]
        state: all-gathered combined/local/flatten state_dict
        singleton_state: all-gathered singleton_state (dimensionless tensors)
        uncollected_opt_state: non-tensor and not-gathered state
        original_sd: the original rank 0's sd

    Returns:
        dict: an unflattened, nonsharded optimizer state, as if FSDP was not there.
    """
    assert all((len(s) == len(instance_list) for s in world_pad_info))
    assert all((len(s[0]) == 1 for s in world_pad_info))
    for local_id, v in uncollected_opt_state.items():
        assert local_id not in state
        state[local_id] = {buffer_name: [x] for buffer_name, x in v.items() if not is_singleton_tensor(x)}
        singleton_state[local_id] = {buffer_name: [x] for buffer_name, x in v.items() if is_singleton_tensor(x)}
    unflat_state, global_to_local_id = _unflatten_optim_state(state, instance_list, world_pad_info, singleton_state)
    param_groups = copy.deepcopy(original_sd['param_groups'])
    num_params = sum([cast(int, m.num_params_managed) for m in instance_list])
    param_groups[0]['params'] = list(range(num_params))
    original_sd['state'] = dict(sorted(unflat_state.items()))
    original_sd['param_id_map'] = global_to_local_id
    original_sd['param_groups'] = param_groups
    original_sd['uncollected_local_ids'] = list(uncollected_opt_state.keys())
    return original_sd