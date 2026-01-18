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
def test_console_read_with_error(caplog):
    msg = "Doesn't work."

    def f(prompt):
        raise Exception(msg)
    with utils.obj_in_module(callbacks, 'consoleread', f), caplog.at_level(logging.ERROR, logger='callbacks.logger'):
        caplog.clear()
        prompt = openrlib.ffi.new('char []', b'foo')
        n = 1000
        buf = openrlib.ffi.new('char [%i]' % n)
        res = callbacks._consoleread(prompt, buf, n, 0)
        assert res == 0
        assert len(caplog.record_tuples) > 0
        for x in caplog.record_tuples:
            assert x == ('rpy2.rinterface_lib.callbacks', logging.ERROR, callbacks._READCONSOLE_EXCEPTION_LOG % msg)