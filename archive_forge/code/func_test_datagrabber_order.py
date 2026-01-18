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
def test_datagrabber_order(tmpdir):
    for file_name in ['sub002_L1_R1.q', 'sub002_L1_R2.q', 'sub002_L2_R1.q', 'sub002_L2_R2.qd', 'sub002_L3_R10.q', 'sub002_L3_R2.q']:
        tmpdir.join(file_name).open('a').close()
    dg = nio.DataGrabber(infields=['sid'])
    dg.inputs.base_directory = tmpdir.strpath
    dg.inputs.template = '%s_L%d_R*.q*'
    dg.inputs.template_args = {'outfiles': [['sid', 1], ['sid', 2], ['sid', 3]]}
    dg.inputs.sid = 'sub002'
    dg.inputs.sort_filelist = True
    res = dg.run()
    outfiles = res.outputs.outfiles
    assert 'sub002_L1_R1' in outfiles[0][0]
    assert 'sub002_L1_R2' in outfiles[0][1]
    assert 'sub002_L2_R1' in outfiles[1][0]
    assert 'sub002_L2_R2' in outfiles[1][1]
    assert 'sub002_L3_R2' in outfiles[2][0]
    assert 'sub002_L3_R10' in outfiles[2][1]