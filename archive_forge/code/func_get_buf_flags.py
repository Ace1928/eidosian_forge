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
def get_buf_flags(specs):
    is_c_contig, is_f_contig = is_cf_contig(specs)
    if is_c_contig:
        return memview_c_contiguous
    elif is_f_contig:
        return memview_f_contiguous
    access, packing = zip(*specs)
    if 'full' in access or 'ptr' in access:
        return memview_full_access
    else:
        return memview_strided_access