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
def get_is_contig_utility(contig_type, ndim):
    assert contig_type in ('C', 'F')
    C = dict(context, ndim=ndim, contig_type=contig_type)
    utility = load_memview_c_utility('MemviewSliceCheckContig', C, requires=[is_contig_utility])
    return utility