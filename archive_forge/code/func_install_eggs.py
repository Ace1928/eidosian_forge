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
def install_eggs(self, spec, dist_filename, tmpdir):
    installer_map = {'.egg': self.install_egg, '.exe': self.install_exe, '.whl': self.install_wheel}
    try:
        install_dist = installer_map[dist_filename.lower()[-4:]]
    except KeyError:
        pass
    else:
        return [install_dist(dist_filename, tmpdir)]
    setup_base = tmpdir
    if os.path.isfile(dist_filename) and (not dist_filename.endswith('.py')):
        unpack_archive(dist_filename, tmpdir, self.unpack_progress)
    elif os.path.isdir(dist_filename):
        setup_base = os.path.abspath(dist_filename)
    if setup_base.startswith(tmpdir) and self.build_directory and (spec is not None):
        setup_base = self.maybe_move(spec, dist_filename, setup_base)
    setup_script = os.path.join(setup_base, 'setup.py')
    if not os.path.exists(setup_script):
        setups = glob(os.path.join(setup_base, '*', 'setup.py'))
        if not setups:
            raise DistutilsError("Couldn't find a setup script in %s" % os.path.abspath(dist_filename))
        if len(setups) > 1:
            raise DistutilsError('Multiple setup scripts in %s' % os.path.abspath(dist_filename))
        setup_script = setups[0]
    if self.editable:
        log.info(self.report_editable(spec, setup_script))
        return []
    else:
        return self.build_and_install(setup_script, setup_base)