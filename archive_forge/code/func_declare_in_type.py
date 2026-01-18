from __future__ import absolute_import
from .StringEncoding import EncodedString
from .Symtab import BuiltinScope, StructOrUnionScope, ModuleScope, Entry
from .Code import UtilityCode, TempitaUtilityCode
from .TypeSlots import Signature
from . import PyrexTypes
def declare_in_type(self, self_type):
    self_type.scope.declare_cproperty(self.py_name, self.property_type, self.call_cname, exception_value=self.exception_value, exception_check=self.exception_check, utility_code=self.utility_code)