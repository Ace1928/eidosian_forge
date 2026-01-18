import torch
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar, Union, overload
from ._functions import Scatter, Gather
import warnings
def scatter_kwargs(inputs: Tuple[Any, ...], kwargs: Optional[Dict[str, Any]], target_gpus: Sequence[Union[int, torch.device]], dim: int=0) -> Tuple[Tuple[Any, ...], Tuple[Dict[str, Any], ...]]:
    """Scatter with support for kwargs dictionary."""
    scattered_inputs = scatter(inputs, target_gpus, dim) if inputs else []
    scattered_kwargs = scatter(kwargs, target_gpus, dim) if kwargs else []
    if len(scattered_inputs) < len(scattered_kwargs):
        scattered_inputs.extend((() for _ in range(len(scattered_kwargs) - len(scattered_inputs))))
    elif len(scattered_kwargs) < len(inputs):
        scattered_kwargs.extend(({} for _ in range(len(scattered_inputs) - len(scattered_kwargs))))
    return (tuple(scattered_inputs), tuple(scattered_kwargs))