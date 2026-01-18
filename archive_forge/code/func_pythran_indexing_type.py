from __future__ import absolute_import
from .PyrexTypes import CType, CTypedefType, CStructOrUnionType
import cython
def pythran_indexing_type(type_, indices):
    return type_remove_ref('decltype(std::declval<%s>()%s)' % (pythran_type(type_), _index_access(_index_type_code, indices)))