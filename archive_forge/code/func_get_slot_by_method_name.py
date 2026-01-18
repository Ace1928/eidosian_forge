from __future__ import absolute_import
from . import Naming
from . import PyrexTypes
from .Errors import error
import copy
def get_slot_by_method_name(self, method_name):
    return self._get_slot_by_method_name(method_name)