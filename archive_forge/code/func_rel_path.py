import os
import re
import sys
import copy
import glob
import atexit
import tempfile
import subprocess
import shutil
import multiprocessing
import textwrap
import importlib.util
from threading import local as tlocal
from functools import reduce
import distutils
from distutils.errors import DistutilsError
def rel_path(path, parent_path):
    """Return path relative to parent_path."""
    pd = os.path.realpath(os.path.abspath(parent_path))
    apath = os.path.realpath(os.path.abspath(path))
    if len(apath) < len(pd):
        return path
    if apath == pd:
        return ''
    if pd == apath[:len(pd)]:
        assert apath[len(pd)] in [os.sep], repr((path, apath[len(pd)]))
        path = apath[len(pd) + 1:]
    return path