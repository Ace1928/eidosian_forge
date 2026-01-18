from __future__ import absolute_import
from . import Naming
from . import PyrexTypes
from .Errors import error
import copy
@property
def fastvar(self):
    if self.use_fastcall:
        return 'FASTCALL'
    else:
        return 'VARARGS'