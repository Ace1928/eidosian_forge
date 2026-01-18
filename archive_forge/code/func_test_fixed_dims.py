from __future__ import absolute_import, division, print_function
import unittest
import pytest
from datashader import datashape
from datashader.datashape.util.testing import assert_dshape_equal
from datashader.datashape.parser import parse
from datashader.datashape import coretypes as ct
from datashader.datashape import DataShapeSyntaxError
def test_fixed_dims(self):
    self.assertEqual(parse('3 * bool', self.sym), ct.DataShape(ct.Fixed(3), ct.bool_))
    self.assertEqual(parse('7 * 3 * bool', self.sym), ct.DataShape(ct.Fixed(7), ct.Fixed(3), ct.bool_))
    self.assertEqual(parse('5 * 3 * 12 * bool', self.sym), ct.DataShape(ct.Fixed(5), ct.Fixed(3), ct.Fixed(12), ct.bool_))
    self.assertEqual(parse('2 * 3 * 4 * 5 * bool', self.sym), ct.DataShape(ct.Fixed(2), ct.Fixed(3), ct.Fixed(4), ct.Fixed(5), ct.bool_))