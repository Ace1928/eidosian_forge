import collections
import functools
import inspect
from typing import overload, Any, Callable, Mapping, Tuple, TypeVar, Type, Sequence, Union
from absl import flags
def as_parsed(*args, **kwargs):
    """Overrides flags by parsing strings, saves flag state similar to flagsaver.

  This function can be used as either a decorator or context manager similar to
  flagsaver.flagsaver(). However, where flagsaver.flagsaver() directly sets the
  flags to new values, this function will parse the provided arguments as if
  they were provided on the command line. Among other things, this will cause
  `FLAGS['flag_name'].present == True`.

  A note on unparsed input: For many flag types, the unparsed version will be
  a single string. However for multi_x (multi_string, multi_integer, multi_enum)
  the unparsed version will be a Sequence of strings.

  Args:
    *args: Tuples of FlagHolders and their unparsed value.
    **kwargs: The keyword args are flag names, and the values are unparsed
      values.

  Returns:
    _ParsingFlagOverrider that serves as a context manager or decorator. Will
    save previous flag state and parse new flags, then on cleanup it will
    restore the previous flag state.
  """
    return _construct_overrider(_ParsingFlagOverrider, *args, **kwargs)