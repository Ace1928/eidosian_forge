import sys
import os
import warnings
import pytest
from io import BytesIO
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy.lib import format
def test_large_header():
    s = BytesIO()
    d = {'shape': tuple(), 'fortran_order': False, 'descr': '<i8'}
    format.write_array_header_1_0(s, d)
    s = BytesIO()
    d['descr'] = [('x' * 256 * 256, '<i8')]
    assert_raises(ValueError, format.write_array_header_1_0, s, d)