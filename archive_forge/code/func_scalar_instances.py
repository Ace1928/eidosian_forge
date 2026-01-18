from itertools import permutations, product
import pytest
from pytest import param
import numpy as np
from numpy.core._rational_tests import rational
from numpy.core._multiarray_umath import _discover_array_parameters
from numpy.testing import (
def scalar_instances(times=True, extended_precision=True, user_dtype=True):
    yield param(np.sqrt(np.float16(5)), id='float16')
    yield param(np.sqrt(np.float32(5)), id='float32')
    yield param(np.sqrt(np.float64(5)), id='float64')
    if extended_precision:
        yield param(np.sqrt(np.longdouble(5)), id='longdouble')
    yield param(np.sqrt(np.complex64(2 + 3j)), id='complex64')
    yield param(np.sqrt(np.complex128(2 + 3j)), id='complex128')
    if extended_precision:
        yield param(np.sqrt(np.longcomplex(2 + 3j)), id='clongdouble')
    yield param(np.int8(2), id='int8')
    yield param(np.int16(2), id='int16')
    yield param(np.int32(2), id='int32')
    yield param(np.int64(2), id='int64')
    yield param(np.uint8(2), id='uint8')
    yield param(np.uint16(2), id='uint16')
    yield param(np.uint32(2), id='uint32')
    yield param(np.uint64(2), id='uint64')
    if user_dtype:
        yield param(rational(1, 2), id='rational')
    structured = np.array([(1, 3)], 'i,i')[0]
    assert isinstance(structured, np.void)
    assert structured.dtype == np.dtype('i,i')
    yield param(structured, id='structured')
    if times:
        yield param(np.timedelta64(2), id='timedelta64[generic]')
        yield param(np.timedelta64(23, 's'), id='timedelta64[s]')
        yield param(np.timedelta64('NaT', 's'), id='timedelta64[s](NaT)')
        yield param(np.datetime64('NaT'), id='datetime64[generic](NaT)')
        yield param(np.datetime64('2020-06-07 12:43', 'ms'), id='datetime64[ms]')
    yield param(np.bytes_(b'1234'), id='bytes')
    yield param(np.str_('2345'), id='unicode')
    yield param(np.void(b'4321'), id='unstructured_void')