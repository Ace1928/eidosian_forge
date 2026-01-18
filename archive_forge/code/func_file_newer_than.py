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
def file_newer_than(path, time):
    ftime = modification_time(path)
    return ftime > time