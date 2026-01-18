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
def slice_iter(slice_type, slice_result, ndim, code, force_strided=False):
    if (slice_type.is_c_contig or slice_type.is_f_contig) and (not force_strided):
        return ContigSliceIter(slice_type, slice_result, ndim, code)
    else:
        return StridedSliceIter(slice_type, slice_result, ndim, code)