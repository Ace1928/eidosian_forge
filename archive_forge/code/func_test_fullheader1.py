import datetime
import os
import sys
from os.path import join as pjoin
from io import StringIO
import numpy as np
from numpy.testing import (assert_array_almost_equal,
from pytest import raises as assert_raises
from scipy.io.arff import loadarff
from scipy.io.arff._arffread import read_header, ParseArffError
def test_fullheader1(self):
    with open(test1) as ofile:
        rel, attrs = read_header(ofile)
    assert_(rel == 'test1')
    assert_(len(attrs) == 5)
    for i in range(4):
        assert_(attrs[i].name == 'attr%d' % i)
        assert_(attrs[i].type_name == 'numeric')
    assert_(attrs[4].name == 'class')
    assert_(attrs[4].values == ('class0', 'class1', 'class2', 'class3'))