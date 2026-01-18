from __future__ import absolute_import
from .PyrexTypes import CType, CTypedefType, CStructOrUnionType
import cython
def pythran_is_numpy_func_supported(func):
    CurF = pythran.tables.MODULES['numpy']
    FL = np_func_to_list(func)
    for F in FL:
        CurF = CurF.get(F, None)
        if CurF is None:
            return False
    return True