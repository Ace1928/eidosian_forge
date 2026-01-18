from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import types
from absl.flags import _argument_parser
from absl.flags import _exceptions
from absl.flags import _flag
from absl.flags import _flagvalues
from absl.flags import _helpers
from absl.flags import _validators
def adopt_module_key_flags(module, flag_values=_flagvalues.FLAGS):
    """Declares that all flags key to a module are key to the current module.

  Args:
    module: module, the module object from which all key flags will be declared
      as key flags to the current module.
    flag_values: FlagValues, the FlagValues instance in which the flags will be
      declared as key flags. This should almost never need to be overridden.

  Raises:
    Error: Raised when given an argument that is a module name (a string),
        instead of a module object.
  """
    if not isinstance(module, types.ModuleType):
        raise _exceptions.Error('Expected a module object, not %r.' % (module,))
    _internal_declare_key_flags([f.name for f in flag_values.get_key_flags_for_module(module.__name__)], flag_values=flag_values)
    if module == _helpers.FLAGS_MODULE:
        _internal_declare_key_flags([_helpers.SPECIAL_FLAGS[name].name for name in _helpers.SPECIAL_FLAGS], flag_values=_helpers.SPECIAL_FLAGS, key_flag_values=flag_values)