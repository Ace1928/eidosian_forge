import os
import copy
import simplejson
import glob
import os.path as op
from subprocess import Popen
import hashlib
from collections import namedtuple
import pytest
import nipype
import nipype.interfaces.io as nio
from nipype.interfaces.base.traits_extension import isdefined
from nipype.interfaces.base import Undefined, TraitError
from nipype.utils.filemanip import dist_is_editable
from subprocess import check_call, CalledProcessError
@pytest.mark.skipif(not have_pybids, reason='Pybids is not installed')
@pytest.mark.skipif(not dist_is_editable('pybids'), reason='Pybids is not installed in editable mode')
def test_bids_fields(tmpdir):
    tmpdir.chdir()
    bg = nio.BIDSDataGrabber(infields=['subject'], outfields=['dwi'])
    bg.inputs.base_dir = os.path.join(datadir, 'ds005')
    bg.inputs.subject = '01'
    bg.inputs.output_query['dwi'] = dict(datatype='dwi')
    results = bg.run()
    assert 'sub-01_dwi.nii.gz' in map(os.path.basename, results.outputs.dwi)