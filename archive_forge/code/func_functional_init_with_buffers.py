import copy
from typing import (
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils._named_member_accessor import NamedMemberAccessor
def functional_init_with_buffers(model_class: Type[nn.Module], ensemble_shape: Union[Tuple[()], Tuple[int]]=(), device: torch.types.Device='cpu'):

    def wrapped(*args, **kwargs):
        if len(ensemble_shape) >= 2:
            raise ValueError('NYI: ensemble_shape with more than 1 element')
        if len(ensemble_shape) == 0:
            model = model_class(*args, **kwargs).to(device)
            return make_functional_deprecated_v1(model)
        num_models = ensemble_shape[0]
        if num_models <= 0:
            raise ValueError(f'num_models {num_models} should be > 0')
        models = tuple((model_class(*args, **kwargs).to(device) for _ in range(num_models)))
        _, _, fn, weight_names, buffer_names = make_functional_with_buffers_deprecated_v1(model_class(*args, **kwargs))
        weights, buffers = zip(*tuple((make_functional_with_buffers_deprecated_v1(model)[:2] for model in models)))
        weights = tuple(zip(*weights))
        weights = tuple((torch.stack(shards).detach() for shards in weights))
        buffers = tuple(zip(*buffers))
        buffers = tuple((torch.stack(shards).detach() for shards in buffers))
        return (weights, buffers, fn, weight_names, buffer_names)
    return wrapped