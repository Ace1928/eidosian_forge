import contextlib
import shutil
import tempfile
from pathlib import Path
import numpy
import pytest
from thinc.api import ArgsKwargs, Linear, Padded, Ragged
from thinc.util import has_cupy, is_cupy_array, is_numpy_array
def assert_lists_match(X, Y):
    assert isinstance(X, list)
    assert isinstance(Y, list)
    assert len(X) == len(Y)
    for x, y in zip(X, Y):
        assert_arrays_match(x, y)
    return True