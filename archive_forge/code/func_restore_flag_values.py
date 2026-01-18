import collections
import functools
import inspect
from typing import overload, Any, Callable, Mapping, Tuple, TypeVar, Type, Sequence, Union
from absl import flags
def restore_flag_values(saved_flag_values: Mapping[str, Mapping[str, Any]], flag_values: flags.FlagValues=FLAGS):
    """Restores flag values based on the dictionary of flag values.

  Args:
    saved_flag_values: {'flag_name': value_dict, ...}
    flag_values: FlagValues, the FlagValues instance from which the flag will be
      restored. This should almost never need to be overridden.
  """
    new_flag_names = list(flag_values)
    for name in new_flag_names:
        saved = saved_flag_values.get(name)
        if saved is None:
            delattr(flag_values, name)
        else:
            if flag_values[name].value != saved['_value']:
                flag_values[name].value = saved['_value']
            flag_values[name].__dict__ = saved