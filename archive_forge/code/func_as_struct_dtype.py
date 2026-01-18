import collections
import ctypes
import re
import numpy as np
from numba.core import errors, types
from numba.core.typing.templates import signature
from numba.np import npdatetime_helpers
from numba.core.errors import TypingError
from numba.core.cgutils import is_nonelike   # noqa: F401
def as_struct_dtype(rec):
    """Convert Numba Record type to NumPy structured dtype
    """
    assert isinstance(rec, types.Record)
    names = []
    formats = []
    offsets = []
    titles = []
    for k, t in rec.members:
        if not rec.is_title(k):
            names.append(k)
            formats.append(as_dtype(t))
            offsets.append(rec.offset(k))
            titles.append(rec.fields[k].title)
    fields = {'names': names, 'formats': formats, 'offsets': offsets, 'itemsize': rec.size, 'titles': titles}
    _check_struct_alignment(rec, fields)
    return np.dtype(fields, align=rec.aligned)