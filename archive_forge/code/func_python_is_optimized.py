from __future__ import (absolute_import, division,
from future import utils
from future.builtins import str, range, open, int, map, list
import contextlib
import errno
import functools
import gc
import socket
import sys
import os
import platform
import shutil
import warnings
import unittest
import importlib
import re
import subprocess
import time
import fnmatch
import logging.handlers
import struct
import tempfile
def python_is_optimized():
    """Find if Python was built with optimizations."""
    import sysconfig
    cflags = sysconfig.get_config_var('PY_CFLAGS') or ''
    final_opt = ''
    for opt in cflags.split():
        if opt.startswith('-O'):
            final_opt = opt
    return final_opt != '' and final_opt != '-O0'