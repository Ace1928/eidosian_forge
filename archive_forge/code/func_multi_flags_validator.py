from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import warnings
from absl.flags import _exceptions
from absl.flags import _flagvalues
from absl.flags import _validators_classes
def multi_flags_validator(flag_names, message='Flag validation failed', flag_values=_flagvalues.FLAGS):
    """A function decorator for defining a multi-flag validator.

  Registers the decorated function as a validator for flag_names, e.g.

  @flags.multi_flags_validator(['foo', 'bar'])
  def _CheckFooBar(flags_dict):
    ...

  See register_multi_flags_validator() for the specification of checker
  function.

  Args:
    flag_names: [str], a list of the flag names to be checked.
    message: str, error text to be shown to the user if checker returns False.
        If checker raises flags.ValidationError, message from the raised
        error will be shown.
    flag_values: flags.FlagValues, optional FlagValues instance to validate
        against.

  Returns:
    A function decorator that registers its function argument as a validator.

  Raises:
    AttributeError: Raised when a flag is not registered as a valid flag name.
  """

    def decorate(function):
        register_multi_flags_validator(flag_names, function, message=message, flag_values=flag_values)
        return function
    return decorate