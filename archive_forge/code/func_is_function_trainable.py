import atexit
import logging
from functools import partial
from types import FunctionType
from typing import Callable, Optional, Type, Union
import ray
import ray.cloudpickle as pickle
from ray.experimental.internal_kv import (
from ray.tune.error import TuneError
from ray.util.annotations import DeveloperAPI
@DeveloperAPI
def is_function_trainable(trainable: Union[str, Callable, Type]) -> bool:
    """Check if a given trainable is a function trainable.
    Either the trainable has been wrapped as a FunctionTrainable class already,
    or it's still a FunctionType/partial/callable."""
    from ray.tune.trainable import FunctionTrainable
    if isinstance(trainable, str):
        trainable = get_trainable_cls(trainable)
    is_wrapped_func = isinstance(trainable, type) and issubclass(trainable, FunctionTrainable)
    return is_wrapped_func or (not isinstance(trainable, type) and (isinstance(trainable, FunctionType) or isinstance(trainable, partial) or callable(trainable)))