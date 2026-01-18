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
def exe_to_egg(self, dist_filename, egg_tmp):
    """Extract a bdist_wininst to the directories an egg would use"""
    prefixes = get_exe_prefixes(dist_filename)
    to_compile = []
    native_libs = []
    top_level = {}

    def process(src, dst):
        s = src.lower()
        for old, new in prefixes:
            if s.startswith(old):
                src = new + src[len(old):]
                parts = src.split('/')
                dst = os.path.join(egg_tmp, *parts)
                dl = dst.lower()
                if dl.endswith('.pyd') or dl.endswith('.dll'):
                    parts[-1] = bdist_egg.strip_module(parts[-1])
                    top_level[os.path.splitext(parts[0])[0]] = 1
                    native_libs.append(src)
                elif dl.endswith('.py') and old != 'SCRIPTS/':
                    top_level[os.path.splitext(parts[0])[0]] = 1
                    to_compile.append(dst)
                return dst
        if not src.endswith('.pth'):
            log.warn("WARNING: can't process %s", src)
        return None
    unpack_archive(dist_filename, egg_tmp, process)
    stubs = []
    for res in native_libs:
        if res.lower().endswith('.pyd'):
            parts = res.split('/')
            resource = parts[-1]
            parts[-1] = bdist_egg.strip_module(parts[-1]) + '.py'
            pyfile = os.path.join(egg_tmp, *parts)
            to_compile.append(pyfile)
            stubs.append(pyfile)
            bdist_egg.write_stub(resource, pyfile)
    self.byte_compile(to_compile)
    bdist_egg.write_safety_flag(os.path.join(egg_tmp, 'EGG-INFO'), bdist_egg.analyze_egg(egg_tmp, stubs))
    for name in ('top_level', 'native_libs'):
        if locals()[name]:
            txt = os.path.join(egg_tmp, 'EGG-INFO', name + '.txt')
            if not os.path.exists(txt):
                f = open(txt, 'w')
                f.write('\n'.join(locals()[name]) + '\n')
                f.close()