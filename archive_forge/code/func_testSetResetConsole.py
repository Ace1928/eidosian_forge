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
def testSetResetConsole():

    def make_callback():
        reset = 0

        def f():
            nonlocal reset
            reset += 1
        return f
    f = make_callback()
    with utils.obj_in_module(callbacks, 'consolereset', f):
        callbacks._consolereset()
        assert f.__closure__[0].cell_contents == 1