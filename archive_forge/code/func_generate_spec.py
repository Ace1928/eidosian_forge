from __future__ import absolute_import
from . import Naming
from . import PyrexTypes
from .Errors import error
import copy
def generate_spec(self, scope, code):
    for slot in self.sub_slots:
        slot.generate_spec(scope, code)