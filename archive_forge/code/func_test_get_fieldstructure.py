import pytest
import numpy as np
import numpy.ma as ma
from numpy.ma.mrecords import MaskedRecords
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_, assert_raises
from numpy.lib.recfunctions import (
def test_get_fieldstructure(self):
    ndtype = np.dtype([('A', '|S3'), ('B', float)])
    test = get_fieldstructure(ndtype)
    assert_equal(test, {'A': [], 'B': []})
    ndtype = np.dtype([('A', int), ('B', [('BA', float), ('BB', '|S1')])])
    test = get_fieldstructure(ndtype)
    assert_equal(test, {'A': [], 'B': [], 'BA': ['B'], 'BB': ['B']})
    ndtype = np.dtype([('A', int), ('B', [('BA', int), ('BB', [('BBA', int), ('BBB', int)])])])
    test = get_fieldstructure(ndtype)
    control = {'A': [], 'B': [], 'BA': ['B'], 'BB': ['B'], 'BBA': ['B', 'BB'], 'BBB': ['B', 'BB']}
    assert_equal(test, control)
    ndtype = np.dtype([])
    test = get_fieldstructure(ndtype)
    assert_equal(test, {})