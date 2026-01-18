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
def easy_install(self, spec, deps=False):
    with self._tmpdir() as tmpdir:
        if not isinstance(spec, Requirement):
            if URL_SCHEME(spec):
                self.not_editable(spec)
                dl = self.package_index.download(spec, tmpdir)
                return self.install_item(None, dl, tmpdir, deps, True)
            elif os.path.exists(spec):
                self.not_editable(spec)
                return self.install_item(None, spec, tmpdir, deps, True)
            else:
                spec = parse_requirement_arg(spec)
        self.check_editable(spec)
        dist = self.package_index.fetch_distribution(spec, tmpdir, self.upgrade, self.editable, not self.always_copy, self.local_index)
        if dist is None:
            msg = 'Could not find suitable distribution for %r' % spec
            if self.always_copy:
                msg += ' (--always-copy skips system and development eggs)'
            raise DistutilsError(msg)
        elif dist.precedence == DEVELOP_DIST:
            self.process_distribution(spec, dist, deps, 'Using')
            return dist
        else:
            return self.install_item(spec, dist.location, tmpdir, deps)