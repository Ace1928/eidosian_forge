import collections.abc as abc
from dataclasses import dataclass
from math import inf
from typing import Any, Callable, Dict, List, Optional
import torch
import torch.distributed as dist
def calc_grad_norm(parameters: List[torch.nn.Parameter], p: float) -> torch.Tensor:
    """Calculate gradient norm of an iterable of parameters.
    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda par: par.grad is not None, parameters))
    if len(parameters) == 0:
        return torch.tensor(0.0)
    p = float(p)
    if p == inf:
        local_norm = max((par.grad.detach().abs().max() for par in parameters))
    else:
        local_norm = torch.norm(torch.stack([torch.norm(par.grad.detach(), p, dtype=torch.float32) for par in parameters]), p).to(dtype=parameters[0].dtype)
    return local_norm