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
def test_consolewrite_print_error(caplog):
    msg = "Doesn't work."

    def f(x):
        raise Exception(msg)
    with utils.obj_in_module(callbacks, 'consolewrite_print', f), caplog.at_level(logging.ERROR, logger='callbacks.logger'):
        code = rinterface.StrSexpVector(['3'])
        caplog.clear()
        rinterface.baseenv['print'](code)
        assert len(caplog.record_tuples) > 0
        for x in caplog.record_tuples:
            assert x == ('rpy2.rinterface_lib.callbacks', logging.ERROR, callbacks._WRITECONSOLE_EXCEPTION_LOG % msg)