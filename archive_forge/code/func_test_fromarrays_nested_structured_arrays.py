import collections.abc
import textwrap
from io import BytesIO
from os import path
from pathlib import Path
import pytest
import numpy as np
from numpy.testing import (
from numpy.compat import pickle
def test_fromarrays_nested_structured_arrays(self):
    arrays = [np.arange(10), np.ones(10, dtype=[('a', '<u2'), ('b', '<f4')])]
    arr = np.rec.fromarrays(arrays)