from __future__ import absolute_import, division, print_function
import ctypes
import unittest
import pytest
from datashader import datashape
from datashader.datashape import dshape, error, DataShape, Record
@pytest.mark.xfail(reason='implements has not been implemented in the new parser')
def test_constraints_error(self):
    self.assertRaises(error.DataShapeTypeError, dshape, 'A : integral * B : numeric')