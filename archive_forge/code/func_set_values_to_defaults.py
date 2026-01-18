import sys
import optparse
import warnings
from optparse import OptParseError, OptionError, OptionValueError, \
from .module import get_introspection_module
from gi import _gi, PyGIDeprecationWarning
from gi._error import GError
def set_values_to_defaults(self):
    for option in self.option_list:
        default = self.defaults.get(option.dest)
        if isinstance(default, str):
            opt_str = option.get_opt_string()
            self.defaults[option.dest] = option.check_value(opt_str, default)
    self.values = optparse.Values(self.defaults)