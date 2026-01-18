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
def test_bids_infields_outfields(tmpdir):
    tmpdir.chdir()
    infields = ['infield1', 'infield2']
    outfields = ['outfield1', 'outfield2']
    bg = nio.BIDSDataGrabber(infields=infields)
    for outfield in outfields:
        bg.inputs.output_query[outfield] = {'key': 'value'}
    for infield in infields:
        assert infield in bg.inputs.traits()
        assert not isdefined(bg.inputs.get()[infield])
    for outfield in outfields:
        assert outfield in bg._outputs().traits()
    bg = nio.BIDSDataGrabber()
    for outfield in ['T1w', 'bold']:
        assert outfield in bg._outputs().traits()