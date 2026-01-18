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
@pytest.mark.skip(reason='WIP (should be run from worker process).')
def test_cleanup():

    def f(saveact, status, runlast):
        return None
    with utils.obj_in_module(callbacks, 'cleanup', f):
        r_quit = rinterface.baseenv['q']
        with pytest.raises(rinterface.embedded.RRuntimeError):
            r_quit()