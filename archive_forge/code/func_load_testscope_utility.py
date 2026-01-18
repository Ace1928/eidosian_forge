from __future__ import absolute_import
from .Symtab import ModuleScope
from .PyrexTypes import *
from .UtilityCode import CythonUtilityCode
from .Errors import error
from .Scanning import StringSourceDescriptor
from . import MemoryView
from .StringEncoding import EncodedString
def load_testscope_utility(cy_util_name, **kwargs):
    return CythonUtilityCode.load(cy_util_name, 'TestCythonScope.pyx', **kwargs)