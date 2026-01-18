from typing import Dict, List, Optional, Sequence, SupportsFloat, Tuple, Type, Union
import numpy as np
import gym.error
from gym import logger
from gym.spaces.space import Space
def get_inf(dtype, sign: str) -> SupportsFloat:
    """Returns an infinite that doesn't break things.

    Args:
        dtype: An `np.dtype`
        sign (str): must be either `"+"` or `"-"`

    Returns:
        Gets an infinite value with the sign and dtype

    Raises:
        TypeError: Unknown sign, use either '+' or '-'
        ValueError: Unknown dtype for infinite bounds
    """
    if np.dtype(dtype).kind == 'f':
        if sign == '+':
            return np.inf
        elif sign == '-':
            return -np.inf
        else:
            raise TypeError(f"Unknown sign {sign}, use either '+' or '-'")
    elif np.dtype(dtype).kind == 'i':
        if sign == '+':
            return np.iinfo(dtype).max - 2
        elif sign == '-':
            return np.iinfo(dtype).min + 2
        else:
            raise TypeError(f"Unknown sign {sign}, use either '+' or '-'")
    else:
        raise ValueError(f'Unknown dtype {dtype} for infinite bounds')