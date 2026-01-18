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
@script_test
def test_nib_tck2trk():
    anat = pjoin(DATA_PATH, 'standard.nii.gz')
    standard_tck = pjoin(DATA_PATH, 'standard.tck')
    with InTemporaryDirectory() as tmpdir:
        shutil.copy(standard_tck, tmpdir)
        standard_trk = pjoin(tmpdir, 'standard.trk')
        standard_tck = pjoin(tmpdir, 'standard.tck')
        cmd = ['nib-tck2trk', standard_tck, anat]
        code, stdout, stderr = run_command(cmd, check_code=False)
        assert code == 2
        assert 'Expecting anatomical image as first argument' in stderr
        cmd = ['nib-tck2trk', anat, standard_tck]
        code, stdout, stderr = run_command(cmd)
        assert len(stdout) == 0
        assert os.path.isfile(standard_trk)
        tck = nib.streamlines.load(standard_tck)
        trk = nib.streamlines.load(standard_trk)
        assert (trk.streamlines.get_data() == tck.streamlines.get_data()).all()
        assert isinstance(trk, nib.streamlines.TrkFile)
        cmd = ['nib-tck2trk', anat, standard_trk]
        code, stdout, stderr = run_command(cmd)
        assert 'Skipping non TCK file' in stdout
        cmd = ['nib-tck2trk', anat, standard_tck]
        code, stdout, stderr = run_command(cmd)
        assert 'Skipping existing file' in stdout
        cmd = ['nib-tck2trk', '--force', anat, standard_tck, standard_tck]
        code, stdout, stderr = run_command(cmd)
        assert len(stdout) == 0
        tck = nib.streamlines.load(standard_tck)
        trk = nib.streamlines.load(standard_trk)
        assert (tck.streamlines.get_data() == trk.streamlines.get_data()).all()