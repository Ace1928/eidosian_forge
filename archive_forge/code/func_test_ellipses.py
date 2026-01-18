from __future__ import absolute_import, division, print_function
import unittest
import pytest
from datashader import datashape
from datashader.datashape.util.testing import assert_dshape_equal
from datashader.datashape.parser import parse
from datashader.datashape import coretypes as ct
from datashader.datashape import DataShapeSyntaxError
def test_ellipses(self):
    self.assertEqual(parse('... * bool', self.sym), ct.DataShape(ct.Ellipsis(), ct.bool_))
    self.assertEqual(parse('M * ... * bool', self.sym), ct.DataShape(ct.TypeVar('M'), ct.Ellipsis(), ct.bool_))
    self.assertEqual(parse('M * ... * 3 * bool', self.sym), ct.DataShape(ct.TypeVar('M'), ct.Ellipsis(), ct.Fixed(3), ct.bool_))