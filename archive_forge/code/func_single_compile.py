import os
import re
import sys
import platform
import shlex
import time
import subprocess
from copy import copy
from pathlib import Path
from distutils import ccompiler
from distutils.ccompiler import (
from distutils.errors import (
from distutils.sysconfig import customize_compiler
from distutils.version import LooseVersion
from numpy.distutils import log
from numpy.distutils.exec_command import (
from numpy.distutils.misc_util import cyg2win32, is_sequence, mingw32, \
import threading
def single_compile(args):
    obj, (src, ext) = args
    if not _needs_build(obj, cc_args, extra_postargs, pp_opts):
        return
    while True:
        with _global_lock:
            if obj not in _processing_files:
                _processing_files.add(obj)
                break
        time.sleep(0.1)
    try:
        with _job_semaphore:
            self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)
    finally:
        with _global_lock:
            _processing_files.remove(obj)