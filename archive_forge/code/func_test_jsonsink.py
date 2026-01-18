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
@pytest.mark.parametrize('inputs_attributes', [{'new_entry': 'someValue'}, {'new_entry': 'someValue', 'test': 'testInfields'}])
def test_jsonsink(tmpdir, inputs_attributes):
    tmpdir.chdir()
    js = nio.JSONFileSink(infields=['test'], in_dict={'foo': 'var'})
    setattr(js.inputs, 'contrasts.alt', 'someNestedValue')
    expected_data = {'contrasts': {'alt': 'someNestedValue'}, 'foo': 'var'}
    for key, val in inputs_attributes.items():
        setattr(js.inputs, key, val)
        expected_data[key] = val
    res = js.run()
    with open(res.outputs.out_file, 'r') as f:
        data = simplejson.load(f)
    assert data == expected_data