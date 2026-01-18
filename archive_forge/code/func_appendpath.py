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
def appendpath(prefix, path):
    if os.path.sep != '/':
        prefix = prefix.replace('/', os.path.sep)
        path = path.replace('/', os.path.sep)
    drive = ''
    if os.path.isabs(path):
        drive = os.path.splitdrive(prefix)[0]
        absprefix = os.path.splitdrive(os.path.abspath(prefix))[1]
        pathdrive, path = os.path.splitdrive(path)
        d = os.path.commonprefix([absprefix, path])
        if os.path.join(absprefix[:len(d)], absprefix[len(d):]) != absprefix or os.path.join(path[:len(d)], path[len(d):]) != path:
            d = os.path.dirname(d)
        subpath = path[len(d):]
        if os.path.isabs(subpath):
            subpath = subpath[1:]
    else:
        subpath = path
    return os.path.normpath(njoin(drive + prefix, subpath))