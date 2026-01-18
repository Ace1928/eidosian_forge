from typing import Dict, Callable
from ray.util.annotations import PublicAPI
from ray.tune.stopper.stopper import Stopper
@classmethod
def is_valid_function(cls, fn):
    is_function = callable(fn) and (not issubclass(type(fn), Stopper))
    if is_function and hasattr(fn, 'stop_all'):
        raise ValueError('Stop object must be ray.tune.Stopper subclass to be detected correctly.')
    return is_function