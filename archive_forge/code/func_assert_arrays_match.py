import contextlib
import shutil
import tempfile
from pathlib import Path
import numpy
import pytest
from thinc.api import ArgsKwargs, Linear, Padded, Ragged
from thinc.util import has_cupy, is_cupy_array, is_numpy_array
def assert_arrays_match(X, Y):
    assert X.dtype == Y.dtype
    assert X.shape[0] == Y.shape[0]
    return True