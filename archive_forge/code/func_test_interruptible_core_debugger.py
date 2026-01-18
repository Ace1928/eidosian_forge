import builtins
import os
import sys
import platform
from tempfile import NamedTemporaryFile
from textwrap import dedent
from unittest.mock import patch
from IPython.core import debugger
from IPython.testing import IPYTHON_TESTING_TIMEOUT_SCALE
from IPython.testing.decorators import skip_win32
import pytest
def test_interruptible_core_debugger():
    """The debugger can be interrupted.

    The presumption is there is some mechanism that causes a KeyboardInterrupt
    (this is implemented in ipykernel).  We want to ensure the
    KeyboardInterrupt cause debugging to cease.
    """

    def raising_input(msg='', called=[0]):
        called[0] += 1
        assert called[0] == 1, 'input() should only be called once!'
        raise KeyboardInterrupt()
    tracer_orig = sys.gettrace()
    try:
        with patch.object(builtins, 'input', raising_input):
            debugger.InterruptiblePdb().set_trace()
    finally:
        sys.settrace(tracer_orig)