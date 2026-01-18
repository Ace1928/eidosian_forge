from collections import abc as container_abcs, defaultdict
from copy import deepcopy
from itertools import chain
import torch
import bitsandbytes.functional as F
def load_state_dict(self, state_dict):
    """Load an optimizer state.

        Arguments:
            state_dict (`dict`):
                An optimizer state (should be returned from a call to `state_dict`) to load.
        """
    state_dict = deepcopy(state_dict)
    groups = self.param_groups
    saved_groups = state_dict['param_groups']
    if len(groups) != len(saved_groups):
        raise ValueError('loaded state dict has a different number of parameter groups')
    param_lens = (len(g['params']) for g in groups)
    saved_lens = (len(g['params']) for g in saved_groups)
    if any((p_len != s_len for p_len, s_len in zip(param_lens, saved_lens))):
        raise ValueError("loaded state dict contains a parameter group that doesn't match the size of optimizer's group")
    id_map = {old_id: p for old_id, p in zip(chain.from_iterable((g['params'] for g in saved_groups)), chain.from_iterable((g['params'] for g in groups)))}

    def cast(param, value):
        """Make a deep copy of value, casting all tensors to device of param."""
        if isinstance(value, torch.Tensor):
            if param.is_floating_point() and value.dtype != torch.uint8:
                value = value.to(param.dtype)
            return value
        elif isinstance(value, dict):
            for k, v in value.items():
                if k in self.non_castable_tensor_keys:
                    value[k] = v.to(param.device)
                else:
                    value[k] = cast(param, v)
            return value
        elif isinstance(value, container_abcs.Iterable):
            return type(value)((cast(param, v) for v in value))
        else:
            return value
    state = defaultdict(dict)
    for k, v in state_dict['state'].items():
        if k in id_map:
            param = id_map[k]
            state[param] = cast(param, v)
        else:
            state[k] = v

    def update_group(group, new_group):
        new_group['params'] = group['params']
        return new_group
    param_groups = [update_group(g, ng) for g, ng in zip(groups, saved_groups)]
    self.__setstate__({'state': state, 'param_groups': param_groups})