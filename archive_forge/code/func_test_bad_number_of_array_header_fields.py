from tempfile import mkdtemp
import os
import io
import shutil
import textwrap
import numpy as np
from numpy import array, transpose, pi
from numpy.testing import (assert_equal, assert_allclose,
import pytest
from pytest import raises as assert_raises
import scipy.sparse
import scipy.io._mmio
import scipy.io._fast_matrix_market as fmm
def test_bad_number_of_array_header_fields(self):
    s = '            %%MatrixMarket matrix array real general\n              3  3 999\n            1.0\n            2.0\n            3.0\n            4.0\n            5.0\n            6.0\n            7.0\n            8.0\n            9.0\n            '
    text = textwrap.dedent(s).encode('ascii')
    with pytest.raises(ValueError, match='not of length 2'):
        scipy.io.mmread(io.BytesIO(text))