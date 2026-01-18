import multiprocessing as mp
from collections import OrderedDict
from ctypes import c_bool
from functools import singledispatch
from typing import Union
import numpy as np
from gym.error import CustomSpaceError
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Space, Tuple
@singledispatch
def create_shared_memory(space: Space, n: int=1, ctx=mp) -> Union[dict, tuple, mp.Array]:
    """Create a shared memory object, to be shared across processes.

    This eventually contains the observations from the vectorized environment.

    Args:
        space: Observation space of a single environment in the vectorized environment.
        n: Number of environments in the vectorized environment (i.e. the number of processes).
        ctx: The multiprocess module

    Returns:
        shared_memory for the shared object across processes.

    Raises:
        CustomSpaceError: Space is not a valid :class:`gym.Space` instance
    """
    raise CustomSpaceError(f'Cannot create a shared memory for space with type `{type(space)}`. Shared memory only supports default Gym spaces (e.g. `Box`, `Tuple`, `Dict`, etc...), and does not support custom Gym spaces.')