from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import warnings
from absl.flags import _exceptions
from absl.flags import _flagvalues
from absl.flags import _validators_classes
def mark_flag_as_required(flag_name, flag_values=_flagvalues.FLAGS):
    """Ensures that flag is not None during program execution.

  Registers a flag validator, which will follow usual validator rules.
  Important note: validator will pass for any non-None value, such as False,
  0 (zero), '' (empty string) and so on.

  If your module might be imported by others, and you only wish to make the flag
  required when the module is directly executed, call this method like this:

    if __name__ == '__main__':
      flags.mark_flag_as_required('your_flag_name')
      app.run()

  Args:
    flag_name: str, name of the flag
    flag_values: flags.FlagValues, optional FlagValues instance where the flag
        is defined.
  Raises:
    AttributeError: Raised when flag_name is not registered as a valid flag
        name.
  """
    if flag_values[flag_name].default is not None:
        warnings.warn('Flag --%s has a non-None default value; therefore, mark_flag_as_required will pass even if flag is not specified in the command line!' % flag_name, stacklevel=2)
    register_validator(flag_name, lambda value: value is not None, message='Flag --{} must have a value other than None.'.format(flag_name), flag_values=flag_values)