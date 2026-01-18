import collections.abc
import textwrap
from io import BytesIO
from os import path
from pathlib import Path
import pytest
import numpy as np
from numpy.testing import (
from numpy.compat import pickle
def test_missing_field(self):
    arr = np.zeros((3,), dtype=[('x', int), ('y', int)])
    assert_raises(KeyError, lambda: arr[['nofield']])