from __future__ import absolute_import, division, print_function
import ctypes
import unittest
import pytest
from datashader import datashape
from datashader.datashape import dshape, error, DataShape, Record
def test_ellipsis_error(self):
    self.assertRaises(error.DataShapeSyntaxError, dshape, 'T * ...')
    self.assertRaises(error.DataShapeSyntaxError, dshape, 'T * S...')