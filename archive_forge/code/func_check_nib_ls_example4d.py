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
def check_nib_ls_example4d(opts=[], hdrs_str='', other_str=''):
    fname = pjoin(DATA_PATH, 'example4d.nii.gz')
    expected_re = f' (int16|[<>]i2) \\[128,  96,  24,   2\\] 2.00x2.00x2.20x2000.00  #exts: 2{hdrs_str} sform{other_str}$'
    cmd = ['nib-ls'] + opts + [fname]
    code, stdout, stderr = run_command(cmd)
    assert fname == stdout[:len(fname)]
    assert_re_in(expected_re, stdout[len(fname):])