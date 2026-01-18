import enum
import sys
import types
import typing
from typing import Text, List, Any, TypeVar, Optional, Union, Type, Iterable, overload
from absl.flags import _argument_parser
from absl.flags import _exceptions
from absl.flags import _flag
from absl.flags import _flagvalues
from absl.flags import _helpers
from absl.flags import _validators
def override_value(flag_holder: _flagvalues.FlagHolder[_T], value: _T) -> None:
    """Overrides the value of the provided flag.

  This value takes precedent over the default value and, when called after flag
  parsing, any value provided at the command line.

  Args:
    flag_holder: FlagHolder, the flag to modify.
    value: The new value.

  Raises:
    IllegalFlagValueError: The value did not pass the flag parser or validators.
  """
    fv = flag_holder._flagvalues
    parsed = fv[flag_holder.name]._parse(value)
    if parsed != value:
        raise _exceptions.IllegalFlagValueError('flag %s: parsed value %r not equal to original %r' % (flag_holder.name, parsed, value))
    setattr(fv, flag_holder.name, value)