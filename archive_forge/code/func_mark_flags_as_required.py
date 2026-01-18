from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import warnings
from absl.flags import _exceptions
from absl.flags import _flagvalues
from absl.flags import _validators_classes
def mark_flags_as_required(flag_names, flag_values=_flagvalues.FLAGS):
    """Ensures that flags are not None during program execution.

  If your module might be imported by others, and you only wish to make the flag
  required when the module is directly executed, call this method like this:

    if __name__ == '__main__':
      flags.mark_flags_as_required(['flag1', 'flag2', 'flag3'])
      app.run()

  Args:
    flag_names: Sequence[str], names of the flags.
    flag_values: flags.FlagValues, optional FlagValues instance where the flags
        are defined.
  Raises:
    AttributeError: If any of flag name has not already been defined as a flag.
  """
    for flag_name in flag_names:
        mark_flag_as_required(flag_name, flag_values)