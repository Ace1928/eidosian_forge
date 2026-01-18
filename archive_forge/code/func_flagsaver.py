import collections
import functools
import inspect
from typing import overload, Any, Callable, Mapping, Tuple, TypeVar, Type, Sequence, Union
from absl import flags
def flagsaver(*args, **kwargs):
    """The main flagsaver interface. See module doc for usage."""
    return _construct_overrider(_FlagOverrider, *args, **kwargs)