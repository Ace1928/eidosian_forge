import functools
import logging
from collections import abc
from typing import Union, Mapping, Any, Callable
from ray.rllib.core.models.specs.specs_base import Spec, TypeSpec
from ray.rllib.core.models.specs.specs_dict import SpecDict
from ray.rllib.core.models.specs.typing import SpecType
from ray.rllib.utils.nested_dict import NestedDict
from ray.util.annotations import DeveloperAPI
@DeveloperAPI
def is_output_decorated(obj: object) -> bool:
    """Returns True if the object is decorated with `check_output_specs`."""
    return hasattr(obj, '__checked_output_specs__')