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
def extract_wininst_cfg(dist_filename):
    """Extract configuration data from a bdist_wininst .exe

    Returns a configparser.RawConfigParser, or None
    """
    f = open(dist_filename, 'rb')
    try:
        endrec = zipfile._EndRecData(f)
        if endrec is None:
            return None
        prepended = endrec[9] - endrec[5] - endrec[6]
        if prepended < 12:
            return None
        f.seek(prepended - 12)
        tag, cfglen, bmlen = struct.unpack('<iii', f.read(12))
        if tag not in (305419898, 305419899):
            return None
        f.seek(prepended - (12 + cfglen))
        init = {'version': '', 'target_version': ''}
        cfg = configparser.RawConfigParser(init)
        try:
            part = f.read(cfglen)
            config = part.split(b'\x00', 1)[0]
            config = config.decode(sys.getfilesystemencoding())
            cfg.read_file(io.StringIO(config))
        except configparser.Error:
            return None
        if not cfg.has_section('metadata') or not cfg.has_section('Setup'):
            return None
        return cfg
    finally:
        f.close()