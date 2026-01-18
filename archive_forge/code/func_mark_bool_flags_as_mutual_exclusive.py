from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import warnings
from absl.flags import _exceptions
from absl.flags import _flagvalues
from absl.flags import _validators_classes
def mark_bool_flags_as_mutual_exclusive(flag_names, required=False, flag_values=_flagvalues.FLAGS):
    """Ensures that only one flag among flag_names is True.

  Args:
    flag_names: [str], names of the flags.
    required: bool. If true, exactly one flag must be True. Otherwise, at most
        one flag can be True, and it is valid for all flags to be False.
    flag_values: flags.FlagValues, optional FlagValues instance where the flags
        are defined.
  """
    for flag_name in flag_names:
        if not flag_values[flag_name].boolean:
            raise _exceptions.ValidationError('Flag --{} is not Boolean, which is required for flags used in mark_bool_flags_as_mutual_exclusive.'.format(flag_name))

    def validate_boolean_mutual_exclusion(flags_dict):
        flag_count = sum((bool(val) for val in flags_dict.values()))
        if flag_count == 1 or (not required and flag_count == 0):
            return True
        raise _exceptions.ValidationError('{} one of ({}) must be True.'.format('Exactly' if required else 'At most', ', '.join(flag_names)))
    register_multi_flags_validator(flag_names, validate_boolean_mutual_exclusion, flag_values=flag_values)