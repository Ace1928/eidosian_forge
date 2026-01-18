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
def test_filelike(self):
    with open(test1) as f1:
        data1, meta1 = loadarff(f1)
    with open(test1) as f2:
        data2, meta2 = loadarff(StringIO(f2.read()))
    assert_(data1 == data2)
    assert_(repr(meta1) == repr(meta2))