from __future__ import absolute_import
from . import Naming
from . import PyrexTypes
from .Errors import error
import copy
def get_special_method_signature(self, name):
    slot = self._get_slot_by_method_name(name)
    if slot:
        return slot.signature
    elif name in richcmp_special_methods:
        return ibinaryfunc
    else:
        return None