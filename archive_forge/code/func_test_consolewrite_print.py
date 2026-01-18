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
def test_consolewrite_print():
    tmp_file = io.StringIO()
    stdout = sys.stdout
    sys.stdout = tmp_file
    try:
        callbacks.consolewrite_print('haha')
    finally:
        sys.stdout = stdout
    tmp_file.flush()
    tmp_file.seek(0)
    assert 'haha' == ''.join((s for s in tmp_file)).rstrip()
    tmp_file.close()