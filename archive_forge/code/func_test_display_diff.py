from io import StringIO
from os.path import join as pjoin
import numpy as np
import pytest
import nibabel as nib
from nibabel.cmdline.diff import *
from nibabel.cmdline.utils import *
from nibabel.testing import data_path
def test_display_diff():
    bogus_names = ['hellokitty.nii.gz', 'privettovarish.nii.gz']
    dict_values = {'datatype': [np.array(2, 'uint8'), np.array(4, 'uint8')], 'bitpix': [np.array(8, 'uint8'), np.array(16, 'uint8')]}
    expected_output = 'These files are different.\nField/File     1:hellokitty.nii.gz                                    2:privettovarish.nii.gz                                \ndatatype       2                                                      4                                                      \nbitpix         8                                                      16                                                     \n'
    assert display_diff(bogus_names, dict_values) == expected_output