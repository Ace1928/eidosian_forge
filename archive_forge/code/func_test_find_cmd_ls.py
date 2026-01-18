import sys
import signal
import os
import time
from _thread import interrupt_main  # Py 3
import threading
import pytest
from IPython.utils.process import (find_cmd, FindCmdError, arg_split,
from IPython.utils.capture import capture_output
from IPython.testing import decorators as dec
from IPython.testing import tools as tt
@dec.skip_win32
def test_find_cmd_ls():
    """Make sure we can find the full path to ls."""
    path = find_cmd('ls')
    assert path.endswith('ls')