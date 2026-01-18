from __future__ import absolute_import
import cython
import sys, copy
from itertools import chain
from . import Builtin
from .Errors import error, warning, InternalError, CompileError, CannotSpecialize
from . import Naming
from . import PyrexTypes
from . import TypeSlots
from .PyrexTypes import py_object_type, error_type
from .Symtab import (ModuleScope, LocalScope, ClosureScope, PropertyScope,
from .Code import UtilityCode
from .StringEncoding import EncodedString
from . import Future
from . import Options
from . import DebugFlags
from .Pythran import has_np_pythran, pythran_type, is_pythran_buffer
from ..Utils import add_metaclass, str_to_number
def as_cclass(self):
    """
        Return this node as if it were declared as an extension class
        """
    if self.is_py3_style_class:
        error(self.classobj.pos, 'Python3 style class could not be represented as C class')
        return
    from . import ExprNodes
    return CClassDefNode(self.pos, visibility='private', module_name=None, class_name=self.name, bases=self.bases or ExprNodes.TupleNode(self.pos, args=[]), decorators=self.decorators, body=self.body, in_pxd=False, doc=self.doc)