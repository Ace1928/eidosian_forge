import sys
import operator
import pytest
import ctypes
import gc
import types
from typing import Any
import numpy as np
import numpy.dtypes
from numpy.core._rational_tests import rational
from numpy.core._multiarray_tests import create_custom_field_dtype
from numpy.testing import (
from numpy.compat import pickle
from itertools import permutations
import random
import hypothesis
from hypothesis.extra import numpy as hynp
def check_canonical(self, dtype, canonical):
    """
        Check most properties relevant to "canonical" versions of a dtype,
        which is mainly native byte order for datatypes supporting this.

        The main work is checking structured dtypes with fields, where we
        reproduce most the actual logic used in the C-code.
        """
    assert type(dtype) is type(canonical)
    assert np.can_cast(dtype, canonical, casting='equiv')
    assert np.can_cast(canonical, dtype, casting='equiv')
    assert canonical.isnative
    assert np.result_type(canonical) == canonical
    if not dtype.names:
        assert dtype.flags == canonical.flags
        return
    assert dtype.flags & 16
    assert dtype.fields.keys() == canonical.fields.keys()

    def aligned_offset(offset, alignment):
        return -(-offset // alignment) * alignment
    totalsize = 0
    max_alignment = 1
    for name in dtype.names:
        new_field_descr = canonical.fields[name][0]
        self.check_canonical(dtype.fields[name][0], new_field_descr)
        expected = 27 & new_field_descr.flags
        assert canonical.flags & expected == expected
        if canonical.isalignedstruct:
            totalsize = aligned_offset(totalsize, new_field_descr.alignment)
            max_alignment = max(new_field_descr.alignment, max_alignment)
        assert canonical.fields[name][1] == totalsize
        assert dtype.fields[name][2:] == canonical.fields[name][2:]
        totalsize += new_field_descr.itemsize
    if canonical.isalignedstruct:
        totalsize = aligned_offset(totalsize, max_alignment)
    assert canonical.itemsize == totalsize
    assert canonical.alignment == max_alignment