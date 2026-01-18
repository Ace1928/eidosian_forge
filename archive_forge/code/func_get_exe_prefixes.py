from glob import glob
from distutils.util import get_platform
from distutils.util import convert_path, subst_vars
from distutils.errors import (
from distutils import log, dir_util
from distutils.command.build_scripts import first_line_re
from distutils.spawn import find_executable
from distutils.command import install
import sys
import os
from typing import Dict, List
import zipimport
import shutil
import tempfile
import zipfile
import re
import stat
import random
import textwrap
import warnings
import site
import struct
import contextlib
import subprocess
import shlex
import io
import configparser
import sysconfig
from sysconfig import get_path
from setuptools import Command
from setuptools.sandbox import run_setup
from setuptools.command import setopt
from setuptools.archive_util import unpack_archive
from setuptools.package_index import (
from setuptools.command import bdist_egg, egg_info
from setuptools.warnings import SetuptoolsDeprecationWarning, SetuptoolsWarning
from setuptools.wheel import Wheel
from pkg_resources import (
import pkg_resources
from ..compat import py39, py311
from .._path import ensure_directory
from ..extern.jaraco.text import yield_lines
def get_exe_prefixes(exe_filename):
    """Get exe->egg path translations for a given .exe file"""
    prefixes = [('PURELIB/', ''), ('PLATLIB/pywin32_system32', ''), ('PLATLIB/', ''), ('SCRIPTS/', 'EGG-INFO/scripts/'), ('DATA/lib/site-packages', '')]
    z = zipfile.ZipFile(exe_filename)
    try:
        for info in z.infolist():
            name = info.filename
            parts = name.split('/')
            if len(parts) == 3 and parts[2] == 'PKG-INFO':
                if parts[1].endswith('.egg-info'):
                    prefixes.insert(0, ('/'.join(parts[:2]), 'EGG-INFO/'))
                    break
            if len(parts) != 2 or not name.endswith('.pth'):
                continue
            if name.endswith('-nspkg.pth'):
                continue
            if parts[0].upper() in ('PURELIB', 'PLATLIB'):
                contents = z.read(name).decode()
                for pth in yield_lines(contents):
                    pth = pth.strip().replace('\\', '/')
                    if not pth.startswith('import'):
                        prefixes.append(('%s/%s/' % (parts[0], pth), ''))
    finally:
        z.close()
    prefixes = [(x.lower(), y) for x, y in prefixes]
    prefixes.sort()
    prefixes.reverse()
    return prefixes