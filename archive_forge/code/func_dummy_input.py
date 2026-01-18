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
@pytest.fixture(scope='module')
def dummy_input(request, tmpdir_factory):
    """
    Function to create a dummy file
    """
    input_path = tmpdir_factory.mktemp('input_data').join('datasink_test_s3.txt')
    input_path.write_binary(b'ABCD1234')
    return str(input_path)