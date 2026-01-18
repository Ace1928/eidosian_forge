import os
import warnings
import pytest
from ....utils.filemanip import split_filename
from ... import base as nib
from ...base import traits, Undefined
from ....interfaces import fsl
from ...utility.wrappers import Function
from ....pipeline import Node
from ..specs import get_filecopy_info
@pytest.fixture(scope='module')
def setup_file(request, tmpdir_factory):
    tmp_dir = tmpdir_factory.mktemp('files')
    tmp_infile = tmp_dir.join('foo.txt')
    with tmp_infile.open('w') as fp:
        fp.writelines(['123456789'])
    tmp_dir.chdir()
    return tmp_infile.strpath