from distutils.dir_util import remove_tree, mkpath
from distutils import log
from types import CodeType
import sys
import os
import re
import textwrap
import marshal
from setuptools.extension import Library
from setuptools import Command
from .._path import ensure_directory
from sysconfig import get_path, get_python_version
def analyze_egg(egg_dir, stubs):
    for flag, fn in safety_flags.items():
        if os.path.exists(os.path.join(egg_dir, 'EGG-INFO', fn)):
            return flag
    if not can_scan():
        return False
    safe = True
    for base, dirs, files in walk_egg(egg_dir):
        for name in files:
            if name.endswith('.py') or name.endswith('.pyw'):
                continue
            elif name.endswith('.pyc') or name.endswith('.pyo'):
                safe = scan_module(egg_dir, base, name, stubs) and safe
    return safe