import base64
from collections import OrderedDict
import importlib
import io
import zlib
from typing import Any, Dict, Optional, Sequence, Type, Union
import numpy as np
import ray
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.gym import try_import_gymnasium_and_gym
from ray.rllib.utils.error import NotSerializable
from ray.rllib.utils.spaces.flexdict import FlexDict
from ray.rllib.utils.spaces.repeated import Repeated
from ray.rllib.utils.spaces.simplex import Simplex
@DeveloperAPI
def deserialize_type(module: Union[str, Type], error: bool=False) -> Optional[Union[str, Type]]:
    """Resolves a class path to a class.
    If the given module is already a class, it is returned as is.
    If the given module is a string, it is imported and the class is returned.

    Args:
        module: The classpath (str) or type to resolve.
        error: Whether to throw a ValueError if `module` could not be resolved into
            a class. If False and `module` is not resolvable, returns None.

    Returns:
        The resolved class or `module` (if `error` is False and no resolution possible).

    Raises:
        ValueError: If `error` is True and `module` cannot be resolved.
    """
    if isinstance(module, type):
        return module
    elif isinstance(module, str):
        try:
            module_path, class_name = module.rsplit('.', 1)
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except (ModuleNotFoundError, ImportError, AttributeError, ValueError) as e:
            if error:
                raise ValueError(f'Could not deserialize the given classpath `module={module}` into a valid python class! Make sure you have all necessary pip packages installed and all custom modules are in your `PYTHONPATH` env variable.') from e
    else:
        raise ValueError(f'`module` ({module} must be type or string (classpath)!')
    return module