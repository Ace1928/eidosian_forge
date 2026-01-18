from __future__ import absolute_import
from .PyrexTypes import CType, CTypedefType, CStructOrUnionType
import cython
def pythran_unaryop_type(op, type_):
    return 'decltype(%sstd::declval<%s>())' % (op, pythran_type(type_))