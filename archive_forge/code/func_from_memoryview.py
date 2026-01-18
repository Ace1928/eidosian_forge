import abc
import collections.abc
from collections import OrderedDict
import enum
import itertools
import typing
from rpy2.rinterface_lib import embedded
from rpy2.rinterface_lib import memorymanagement
from rpy2.rinterface_lib import openrlib
import rpy2.rinterface_lib._rinterface_capi as _rinterface
from rpy2.rinterface_lib._rinterface_capi import _evaluated_promise
from rpy2.rinterface_lib._rinterface_capi import SupportsSEXP
from rpy2.rinterface_lib import conversion
from rpy2.rinterface_lib.conversion import _cdata_res_to_rinterface
from rpy2.rinterface_lib import na_values
@classmethod
@_cdata_res_to_rinterface
def from_memoryview(cls, mview: memoryview) -> VT:
    """Create an R vector/array from a memoryview.

        The memoryview must be contiguous, and the C representation
        for the vector must be compatible between R and Python. If
        not the case, a :class:`ValueError` exception with will be
        raised."""
    if not embedded.isready():
        raise embedded.RNotReadyError('Embedded R is not ready to use.')
    if not mview.contiguous:
        raise ValueError('The memory view must be contiguous.')
    if not cls._check_C_compatible(mview):
        cls._raise_incompatible_C_size(mview)
    r_vector = None
    n = len(mview)
    with memorymanagement.rmemory() as rmemory:
        r_vector = rmemory.protect(openrlib.rlib.Rf_allocVector(cls._R_TYPE, n))
        dest_ptr = cls._R_GET_PTR(r_vector)
        src_ptr = _rinterface.ffi.from_buffer(mview)
        nbytes = n * mview.itemsize
        _rinterface.ffi.memmove(dest_ptr, src_ptr, nbytes)
    return r_vector