from __future__ import absolute_import, division, print_function
import unittest
import pytest
from datashader import datashape
from datashader.datashape.util.testing import assert_dshape_equal
from datashader.datashape.parser import parse
from datashader.datashape import coretypes as ct
from datashader.datashape import DataShapeSyntaxError
def assertExpectedParse(ds_str, expected_a, expected_b):
    expected_arg[0] = expected_a
    expected_arg[1] = expected_b
    self.assertEqual(parse(ds_str, sym), ct.DataShape(ct.float32))
    self.assertEqual(expected_arg, [None, None], 'The test binary type constructor did not run')