from __future__ import absolute_import
from .Symtab import ModuleScope
from .PyrexTypes import *
from .UtilityCode import CythonUtilityCode
from .Errors import error
from .Scanning import StringSourceDescriptor
from . import MemoryView
from .StringEncoding import EncodedString
def populate_cython_scope(self):
    type_object = self.declare_typedef('PyTypeObject', base_type=c_void_type, pos=None, cname='PyTypeObject')
    type_object.is_void = True
    type_object_type = type_object.type
    self.declare_cfunction('PyObject_TypeCheck', CFuncType(c_bint_type, [CFuncTypeArg('o', py_object_type, None), CFuncTypeArg('t', c_ptr_type(type_object_type), None)]), pos=None, defining=1, cname='PyObject_TypeCheck')