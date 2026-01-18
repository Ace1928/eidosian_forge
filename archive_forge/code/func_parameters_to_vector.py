import torch
from typing import Iterable, Optional
def parameters_to_vector(parameters: Iterable[torch.Tensor]) -> torch.Tensor:
    """Flatten an iterable of parameters into a single vector.

    Args:
        parameters (Iterable[Tensor]): an iterable of Tensors that are the
            parameters of a model.

    Returns:
        The parameters represented by a single vector
    """
    param_device = None
    vec = []
    for param in parameters:
        param_device = _check_param_device(param, param_device)
        vec.append(param.view(-1))
    return torch.cat(vec)