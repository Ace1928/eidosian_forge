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
def test_datasink():
    ds = nio.DataSink()
    assert ds.inputs.parameterization
    assert ds.inputs.base_directory == Undefined
    assert ds.inputs.strip_dir == Undefined
    assert ds.inputs._outputs == {}
    ds = nio.DataSink(base_directory='foo')
    assert ds.inputs.base_directory == 'foo'
    ds = nio.DataSink(infields=['test'])
    assert 'test' in ds.inputs.copyable_trait_names()