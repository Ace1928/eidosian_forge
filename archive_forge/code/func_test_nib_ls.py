import csv
import os
import shutil
import sys
import unittest
from glob import glob
from os.path import abspath, basename, dirname, exists
from os.path import join as pjoin
from os.path import splitext
import numpy as np
import pytest
from numpy.testing import assert_almost_equal
import nibabel as nib
from ..loadsave import load
from ..orientations import aff2axcodes, inv_ornt_aff
from ..testing import assert_data_similar, assert_dt_equal, assert_re_in
from ..tmpdirs import InTemporaryDirectory
from .nibabel_data import needs_nibabel_data
from .scriptrunner import ScriptRunner
from .test_parrec import DTI_PAR_BVALS, DTI_PAR_BVECS
from .test_parrec import EXAMPLE_IMAGES as PARREC_EXAMPLES
from .test_parrec_data import AFF_OFF, BALLS
@pytest.mark.parametrize('args', [[], [['-H', 'dim,bitpix'], ' \\[  4 128  96  24   2   1   1   1\\] 16'], [['-c'], '', ' !1030 uniques. Use --all-counts'], [['-c', '--all-counts'], '', ' 2:3 3:2 4:1 5:1.*'], [['-c', '-s', '--all-counts'], '', ' \\[229725\\] \\[2, 1.2e\\+03\\] 2:3 3:2 4:1 5:1.*'], [['-c', '-s', '-z', '--all-counts'], '', ' \\[589824\\] \\[0, 1.2e\\+03\\] 0:360099 2:3 3:2 4:1 5:1.*']])
@script_test
def test_nib_ls(args):
    check_nib_ls_example4d(*args)