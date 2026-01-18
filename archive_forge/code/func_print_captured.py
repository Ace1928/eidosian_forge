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
def print_captured(captured, output, header_line=None):
    captured = prepare_captured(captured)
    if captured:
        if header_line:
            output.write(header_line)
        output.write(captured)