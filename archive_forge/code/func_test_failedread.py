import skvideo.io
import sys
import numpy as np
import hashlib
import os
from numpy.testing import assert_equal
from nose.tools import *
@raises(OSError)
def test_failedread():
    skvideo.io.vread('garbage')