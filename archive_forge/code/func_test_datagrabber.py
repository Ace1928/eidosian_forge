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
def test_datagrabber():
    dg = nio.DataGrabber()
    assert dg.inputs.template == Undefined
    assert dg.inputs.base_directory == Undefined
    assert dg.inputs.template_args == {'outfiles': []}