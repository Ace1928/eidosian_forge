import contextlib
import shutil
import tempfile
from pathlib import Path
import numpy
import pytest
from thinc.api import ArgsKwargs, Linear, Padded, Ragged
from thinc.util import has_cupy, is_cupy_array, is_numpy_array
def assert_padded_data_match(X, Y):
    return assert_paddeds_match(Padded(*X), Padded(*Y))