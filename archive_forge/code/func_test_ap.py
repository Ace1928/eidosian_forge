from io import StringIO
from os.path import join as pjoin
import numpy as np
import pytest
import nibabel as nib
from nibabel.cmdline.diff import *
from nibabel.cmdline.utils import *
from nibabel.testing import data_path
def test_ap():
    assert ap([1, 2], '%2d') == ' 1,  2'
    assert ap([1, 2], '%3d') == '  1,   2'
    assert ap([1, 2], '%-2d') == '1 , 2 '
    assert ap([1, 2], '%d', '+') == '1+2'
    assert ap([1, 2, 3], '%d', '-') == '1-2-3'