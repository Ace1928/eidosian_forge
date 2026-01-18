from __future__ import absolute_import
from . import Naming
from . import PyrexTypes
from .Errors import error
import copy
def fixed_arg_type(self, i):
    return self.format_map[self.fixed_arg_format[i]]