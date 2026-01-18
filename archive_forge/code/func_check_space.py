import inspect
from functools import partial
from typing import Callable
import numpy as np
from gym import Space, error, logger, spaces
def check_space(space: Space, space_type: str, check_box_space_fn: Callable[[spaces.Box], None]):
    """A passive check of the environment action space that should not affect the environment."""
    if not isinstance(space, spaces.Space):
        raise AssertionError(f'{space_type} space does not inherit from `gym.spaces.Space`, actual type: {type(space)}')
    elif isinstance(space, spaces.Box):
        check_box_space_fn(space)
    elif isinstance(space, spaces.Discrete):
        assert 0 < space.n, f"Discrete {space_type} space's number of elements must be positive, actual number of elements: {space.n}"
        assert space.shape == (), f"Discrete {space_type} space's shape should be empty, actual shape: {space.shape}"
    elif isinstance(space, spaces.MultiDiscrete):
        assert space.shape == space.nvec.shape, f"Multi-discrete {space_type} space's shape must be equal to the nvec shape, space shape: {space.shape}, nvec shape: {space.nvec.shape}"
        assert np.all(0 < space.nvec), f"Multi-discrete {space_type} space's all nvec elements must be greater than 0, actual nvec: {space.nvec}"
    elif isinstance(space, spaces.MultiBinary):
        assert np.all(0 < np.asarray(space.shape)), f"Multi-binary {space_type} space's all shape elements must be greater than 0, actual shape: {space.shape}"
    elif isinstance(space, spaces.Tuple):
        assert 0 < len(space.spaces), f'An empty Tuple {space_type} space is not allowed.'
        for subspace in space.spaces:
            check_space(subspace, space_type, check_box_space_fn)
    elif isinstance(space, spaces.Dict):
        assert 0 < len(space.spaces.keys()), f'An empty Dict {space_type} space is not allowed.'
        for subspace in space.values():
            check_space(subspace, space_type, check_box_space_fn)