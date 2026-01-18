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
def test_choosefile_error():

    def f(prompt):
        raise Exception("Doesn't work.")
    with utils.obj_in_module(callbacks, 'consolewrite_print', utils.noconsole):
        with utils.obj_in_module(callbacks, 'choosefile', f):
            with pytest.raises(rinterface.embedded.RRuntimeError):
                with pytest.warns(rinterface.RRuntimeWarning):
                    rinterface.baseenv['file.choose']()