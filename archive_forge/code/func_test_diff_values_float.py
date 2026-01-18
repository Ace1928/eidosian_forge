from os.path import abspath, dirname
from os.path import join as pjoin
import numpy as np
from nibabel.cmdline.diff import are_values_different
def test_diff_values_float():
    assert not are_values_different(0.0, 0.0)
    assert not are_values_different(0.0, 0.0, 0.0)
    assert not are_values_different(1.1, 1.1)
    assert are_values_different(0.0, 1.1)
    assert are_values_different(0.0, 0, 1.1)
    assert are_values_different(1.0, 2.0)