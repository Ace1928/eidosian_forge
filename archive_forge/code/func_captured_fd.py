from __future__ import absolute_import
import cython
import os
import sys
import re
import io
import codecs
import glob
import shutil
import tempfile
from functools import wraps
from . import __version__ as cython_version
@try_finally_contextmanager
def captured_fd(stream=2, encoding=None):
    orig_stream = os.dup(stream)
    try:
        with tempfile.TemporaryFile(mode='a+b') as temp_file:

            def read_output(_output=[b'']):
                if not temp_file.closed:
                    temp_file.seek(0)
                    _output[0] = temp_file.read()
                return _output[0]
            os.dup2(temp_file.fileno(), stream)

            def get_output():
                result = read_output()
                return result.decode(encoding) if encoding else result
            yield get_output
            os.dup2(orig_stream, stream)
            read_output()
    finally:
        os.close(orig_stream)