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
def register_trainable(name: str, trainable: Union[Callable, Type], warn: bool=True):
    """Register a trainable function or class.

    This enables a class or function to be accessed on every Ray process
    in the cluster.

    Args:
        name: Name to register.
        trainable: Function or tune.Trainable class. Functions must
            take (config, status_reporter) as arguments and will be
            automatically converted into a class during registration.
    """
    from ray.tune.trainable import wrap_function
    from ray.tune.trainable import Trainable
    if isinstance(trainable, type):
        logger.debug('Detected class for trainable.')
    elif isinstance(trainable, FunctionType) or isinstance(trainable, partial):
        logger.debug('Detected function for trainable.')
        trainable = wrap_function(trainable, warn=warn)
    elif callable(trainable):
        logger.info('Detected unknown callable for trainable. Converting to class.')
        trainable = wrap_function(trainable, warn=warn)
    if not issubclass(trainable, Trainable):
        raise TypeError('Second argument must be convertable to Trainable', trainable)
    _global_registry.register(TRAINABLE_CLASS, name, trainable)