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
def get_cmd(cmdname, _cache={}):
    if cmdname not in _cache:
        import distutils.core
        dist = distutils.core._setup_distribution
        if dist is None:
            from distutils.errors import DistutilsInternalError
            raise DistutilsInternalError('setup distribution instance not initialized')
        cmd = dist.get_command_obj(cmdname)
        _cache[cmdname] = cmd
    return _cache[cmdname]