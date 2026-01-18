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
def test_ExportFile(tmp_path):
    testin = tmp_path / 'in.txt'
    testin.write_text('test string', encoding='utf-8')
    i = nio.ExportFile()
    i.inputs.in_file = str(testin)
    i.inputs.out_file = str(tmp_path / 'out.tsv')
    i.inputs.check_extension = True
    with pytest.raises(RuntimeError):
        i.run()
    i.inputs.check_extension = False
    i.run()
    assert (tmp_path / 'out.tsv').read_text() == 'test string'
    i.inputs.out_file = str(tmp_path / 'out.txt')
    i.inputs.check_extension = True
    i.run()
    assert (tmp_path / 'out.txt').read_text() == 'test string'
    with pytest.raises(FileExistsError):
        i.run()
    i.inputs.clobber = True
    i.run()
    assert (tmp_path / 'out.txt').read_text() == 'test string'