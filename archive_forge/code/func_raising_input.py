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
def raising_input(msg='', called=[0]):
    called[0] += 1
    assert called[0] == 1, 'input() should only be called once!'
    raise KeyboardInterrupt()