from __future__ import absolute_import
from . import Naming
from . import PyrexTypes
from .Errors import error
import copy
def get_slot_code_by_name(scope, slot_name):
    slot = get_slot_by_name(slot_name, scope.directives)
    return slot.slot_code(scope)