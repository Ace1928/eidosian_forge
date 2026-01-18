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
def get_copy_new_utility(pos, from_memview, to_memview):
    if from_memview.dtype != to_memview.dtype and (not (from_memview.dtype.is_cv_qualified and from_memview.dtype.cv_base_type == to_memview.dtype)):
        error(pos, 'dtypes must be the same!')
        return
    if len(from_memview.axes) != len(to_memview.axes):
        error(pos, 'number of dimensions must be same')
        return
    if not (to_memview.is_c_contig or to_memview.is_f_contig):
        error(pos, 'to_memview must be c or f contiguous.')
        return
    for access, packing in from_memview.axes:
        if access != 'direct':
            error(pos, "cannot handle 'full' or 'ptr' access at this time.")
            return
    if to_memview.is_c_contig:
        mode = 'c'
        contig_flag = memview_c_contiguous
    else:
        assert to_memview.is_f_contig
        mode = 'fortran'
        contig_flag = memview_f_contiguous
    return load_memview_c_utility('CopyContentsUtility', context=dict(context, mode=mode, dtype_decl=to_memview.dtype.empty_declaration_code(), contig_flag=contig_flag, ndim=to_memview.ndim, func_cname=copy_c_or_fortran_cname(to_memview), dtype_is_object=int(to_memview.dtype.is_pyobject)), requires=[copy_contents_new_utility])