from __future__ import annotations
import numpy as np
import pytest
from xarray.core import dtypes
@pytest.mark.parametrize('kind, expected', [('a', (np.dtype('O'), 'nan')), ('b', (np.float32, 'nan')), ('B', (np.float32, 'nan')), ('c', (np.dtype('O'), 'nan')), ('D', (np.complex128, '(nan+nanj)')), ('d', (np.float64, 'nan')), ('e', (np.float16, 'nan')), ('F', (np.complex64, '(nan+nanj)')), ('f', (np.float32, 'nan')), ('h', (np.float32, 'nan')), ('H', (np.float32, 'nan')), ('i', (np.float64, 'nan')), ('I', (np.float64, 'nan')), ('l', (np.float64, 'nan')), ('L', (np.float64, 'nan')), ('m', (np.timedelta64, 'NaT')), ('M', (np.datetime64, 'NaT')), ('O', (np.dtype('O'), 'nan')), ('p', (np.float64, 'nan')), ('P', (np.float64, 'nan')), ('q', (np.float64, 'nan')), ('Q', (np.float64, 'nan')), ('S', (np.dtype('O'), 'nan')), ('U', (np.dtype('O'), 'nan')), ('V', (np.dtype('O'), 'nan'))])
def test_maybe_promote(kind, expected) -> None:
    actual = dtypes.maybe_promote(np.dtype(kind))
    assert actual[0] == expected[0]
    assert str(actual[1]) == expected[1]