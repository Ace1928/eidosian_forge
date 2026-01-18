import sys
import os
import warnings
import pytest
from io import BytesIO
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy.lib import format
def test_empty_npz(tmpdir):
    fname = os.path.join(tmpdir, 'nothing.npz')
    np.savez(fname)
    with np.load(fname) as nps:
        pass