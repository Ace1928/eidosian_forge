import collections
import functools
import inspect
from typing import overload, Any, Callable, Mapping, Tuple, TypeVar, Type, Sequence, Union
from absl import flags
def save_flag_values(flag_values: flags.FlagValues=FLAGS) -> Mapping[str, Mapping[str, Any]]:
    """Returns copy of flag values as a dict.

  Args:
    flag_values: FlagValues, the FlagValues instance with which the flag will be
      saved. This should almost never need to be overridden.

  Returns:
    Dictionary mapping keys to values. Keys are flag names, values are
    corresponding ``__dict__`` members. E.g. ``{'key': value_dict, ...}``.
  """
    return {name: _copy_flag_dict(flag_values[name]) for name in flag_values}