from __future__ import absolute_import
from . import Naming
from . import PyrexTypes
from .Errors import error
import copy
def substructure_cname(self, scope):
    return '%s%s_%s' % (Naming.pyrex_prefix, self.slot_name, scope.class_name)