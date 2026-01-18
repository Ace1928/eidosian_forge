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
def check_nib_diff_examples():
    fnames = [pjoin(DATA_PATH, f) for f in ('standard.nii.gz', 'example4d.nii.gz')]
    code, stdout, stderr = run_command(['nib-diff'] + fnames, check_code=False)
    checked_fields = ['Field/File', 'regular', 'dim_info', 'dim', 'datatype', 'bitpix', 'pixdim', 'slice_end', 'xyzt_units', 'cal_max', 'descrip', 'qform_code', 'sform_code', 'quatern_b', 'quatern_c', 'quatern_d', 'qoffset_x', 'qoffset_y', 'qoffset_z', 'srow_x', 'srow_y', 'srow_z', 'DATA(md5)', 'DATA(diff 1:)']
    for item in checked_fields:
        assert item in stdout
    fnames2 = [pjoin(DATA_PATH, f) for f in ('example4d.nii.gz', 'example4d.nii.gz')]
    code, stdout, stderr = run_command(['nib-diff'] + fnames2, check_code=False)
    assert stdout == 'These files are identical.'
    fnames3 = [pjoin(DATA_PATH, f) for f in ('standard.nii.gz', 'example4d.nii.gz', 'example_nifti2.nii.gz')]
    code, stdout, stderr = run_command(['nib-diff'] + fnames3, check_code=False)
    for item in checked_fields:
        assert item in stdout
    fnames4 = [pjoin(DATA_PATH, f) for f in ('standard.nii.gz', 'standard.nii.gz', 'standard.nii.gz')]
    code, stdout, stderr = run_command(['nib-diff'] + fnames4, check_code=False)
    assert stdout == 'These files are identical.'
    code, stdout, stderr = run_command(['nib-diff', '--dt', 'float64'] + fnames, check_code=False)
    for item in checked_fields:
        assert item in stdout