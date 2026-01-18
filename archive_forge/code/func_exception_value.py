from __future__ import absolute_import
from . import Naming
from . import PyrexTypes
from .Errors import error
import copy
def exception_value(self):
    return self.error_value_map.get(self.ret_format)