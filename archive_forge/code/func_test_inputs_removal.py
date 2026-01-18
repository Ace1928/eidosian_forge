import os
from copy import deepcopy
import pytest
from .... import config
from ....interfaces import utility as niu
from ....interfaces import base as nib
from ... import engine as pe
from ..utils import merge_dict
from .test_base import EngineTestInterface
from .test_utils import UtilsTestInterface
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe
import nipype.interfaces.spm as spm
import os
from io import StringIO
from nipype.utils.config import config
def test_inputs_removal(tmpdir):
    file1 = tmpdir.join('file1.txt')
    file1.write('dummy_file')
    n1 = pe.Node(UtilsTestInterface(), base_dir=tmpdir.strpath, name='testinputs')
    n1.inputs.in_file = file1.strpath
    n1.config = {'execution': {'keep_inputs': True}}
    n1.config = merge_dict(deepcopy(config._sections), n1.config)
    n1.run()
    assert tmpdir.join(n1.name, 'file1.txt').check()
    n1.inputs.in_file = file1.strpath
    n1.config = {'execution': {'keep_inputs': False}}
    n1.config = merge_dict(deepcopy(config._sections), n1.config)
    n1.overwrite = True
    n1.run()
    assert not tmpdir.join(n1.name, 'file1.txt').check()