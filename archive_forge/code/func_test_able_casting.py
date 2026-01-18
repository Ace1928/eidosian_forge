import os
from platform import machine
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ..casting import (
from ..testing import suppress_warnings
def test_able_casting():
    types = sctypes['int'] + sctypes['uint']
    for in_type in types:
        in_info = np.iinfo(in_type)
        in_mn, in_mx = (in_info.min, in_info.max)
        A = np.zeros((1,), dtype=in_type)
        for out_type in types:
            out_info = np.iinfo(out_type)
            out_mn, out_mx = (out_info.min, out_info.max)
            B = np.zeros((1,), dtype=out_type)
            ApBt = (A + B).dtype.type
            able_type = able_int_type([in_mn, in_mx, out_mn, out_mx])
            if able_type is None:
                assert ApBt == np.float64
                continue
            assert np.dtype(ApBt).str == np.dtype(able_type).str