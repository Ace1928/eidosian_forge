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
def test_flushconsole_with_error(caplog):
    msg = "Doesn't work."

    def f():
        raise Exception(msg)
    with utils.obj_in_module(callbacks, 'consoleflush', f), caplog.at_level(logging.ERROR, logger='callbacks.logger'):
        caplog.clear()
        rinterface.globalenv.find('flush.console')()
        assert len(caplog.record_tuples) > 0
        for x in caplog.record_tuples:
            assert x == ('rpy2.rinterface_lib.callbacks', logging.ERROR, callbacks._FLUSHCONSOLE_EXCEPTION_LOG % msg)