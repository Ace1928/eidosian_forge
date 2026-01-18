from .. import utils
import builtins
import io
import logging
import os
import pytest
import tempfile
import sys
import rpy2.rinterface as rinterface
from rpy2.rinterface_lib import callbacks
from rpy2.rinterface_lib import openrlib
def test_choosefile_default():
    inputvalue = 'foo'
    with utils.obj_in_module(builtins, 'input', lambda x: inputvalue):
        assert callbacks.choosefile('foo') == inputvalue