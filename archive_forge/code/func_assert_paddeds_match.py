import contextlib
import shutil
import tempfile
from pathlib import Path
import numpy
import pytest
from thinc.api import ArgsKwargs, Linear, Padded, Ragged
from thinc.util import has_cupy, is_cupy_array, is_numpy_array
def assert_paddeds_match(X, Y):
    assert isinstance(X, Padded)
    assert isinstance(Y, Padded)
    assert_arrays_match(X.size_at_t, Y.size_at_t)
    assert assert_arrays_match(X.lengths, Y.lengths)
    assert assert_arrays_match(X.indices, Y.indices)
    assert X.data.dtype == Y.data.dtype
    assert X.data.shape[1] == Y.data.shape[1]
    assert X.data.shape[0] == Y.data.shape[0]
    return True