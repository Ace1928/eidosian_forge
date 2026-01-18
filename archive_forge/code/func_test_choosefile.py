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
@pytest.mark.skipif(os.name == 'nt', reason='Not supported on Windows')
def test_choosefile():
    me = 'me'

    def chooseMe(new):
        return me
    with utils.obj_in_module(callbacks, 'choosefile', chooseMe):
        res = rinterface.baseenv['file.choose']()
        assert me == res[0]