import contextlib
import functools
import inspect
import math
from collections.abc import Iterable
from typing import Callable, Dict, Union, Any
from pennylane import QNode
def init_weight(weight_name: str, weight_size: tuple) -> torch.Tensor:
    """Initialize weights.

            Args:
                weight_name (str): weight name
                weight_size (tuple): size of the weight

            Returns:
                torch.Tensor: tensor containing the weights
            """
    if init_method is None:
        init = functools.partial(torch.nn.init.uniform_, b=2 * math.pi)
    elif callable(init_method):
        init = init_method
    elif isinstance(init_method, dict):
        init = init_method[weight_name]
        if isinstance(init, torch.Tensor):
            if tuple(init.shape) != weight_size:
                raise ValueError(f"The Tensor specified for weight '{weight_name}' doesn't have the " + 'appropiate shape.')
            return init
    return init(torch.Tensor(*weight_size)) if weight_size else init(torch.Tensor(1))[0]