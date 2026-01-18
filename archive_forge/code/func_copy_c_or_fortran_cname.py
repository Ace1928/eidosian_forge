from __future__ import absolute_import
from .Errors import CompileError, error
from . import ExprNodes
from .ExprNodes import IntNode, NameNode, AttributeNode
from . import Options
from .Code import UtilityCode, TempitaUtilityCode
from .UtilityCode import CythonUtilityCode
from . import Buffer
from . import PyrexTypes
from . import ModuleNode
def copy_c_or_fortran_cname(memview):
    if memview.is_c_contig:
        c_or_f = 'c'
    else:
        c_or_f = 'f'
    return '__pyx_memoryview_copy_slice_%s_%s' % (memview.specialization_suffix(), c_or_f)