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
def test_consoleread():
    msg_orig = 'yes'

    def sayyes(prompt):
        return msg_orig
    with utils.obj_in_module(callbacks, 'consoleread', sayyes):
        prompt = openrlib.ffi.new('char []', b'foo')
        n = 1000
        buf = openrlib.ffi.new('char [%i]' % n)
        res = callbacks._consoleread(prompt, buf, n, 0)
    assert res == 1
    msg = openrlib.ffi.string(buf).decode('utf-8')
    assert msg_orig == msg.rstrip()