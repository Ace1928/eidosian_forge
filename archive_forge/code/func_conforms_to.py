from __future__ import absolute_import
import copy
import hashlib
import re
from functools import partial
from itertools import product
from Cython.Utils import cached_function
from .Code import UtilityCode, LazyUtilityCode, TempitaUtilityCode
from . import StringEncoding
from . import Naming
from .Errors import error, CannotSpecialize, performance_hint
def conforms_to(self, dst, broadcast=False, copying=False):
    """
        Returns True if src conforms to dst, False otherwise.

        If conformable, the types are the same, the ndims are equal, and each axis spec is conformable.

        Any packing/access spec is conformable to itself.

        'direct' and 'ptr' are conformable to 'full'.
        'contig' and 'follow' are conformable to 'strided'.
        Any other combo is not conformable.
        """
    from . import MemoryView
    src = self
    src_dtype, dst_dtype = (src.dtype, dst.dtype)
    if not copying:
        if src_dtype.is_const and (not dst_dtype.is_const):
            return False
        if src_dtype.is_volatile and (not dst_dtype.is_volatile):
            return False
    if src_dtype.is_cv_qualified:
        src_dtype = src_dtype.cv_base_type
    if dst_dtype.is_cv_qualified:
        dst_dtype = dst_dtype.cv_base_type
    if not src_dtype.same_as(dst_dtype):
        return False
    if src.ndim != dst.ndim:
        if broadcast:
            src, dst = MemoryView.broadcast_types(src, dst)
        else:
            return False
    for src_spec, dst_spec in zip(src.axes, dst.axes):
        src_access, src_packing = src_spec
        dst_access, dst_packing = dst_spec
        if src_access != dst_access and dst_access != 'full':
            return False
        if src_packing != dst_packing and dst_packing != 'strided' and (not copying):
            return False
    return True