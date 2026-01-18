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
def build_and_install(self, setup_script, setup_base):
    args = ['bdist_egg', '--dist-dir']
    dist_dir = tempfile.mkdtemp(prefix='egg-dist-tmp-', dir=os.path.dirname(setup_script))
    try:
        self._set_fetcher_options(os.path.dirname(setup_script))
        args.append(dist_dir)
        self.run_setup(setup_script, setup_base, args)
        all_eggs = Environment([dist_dir])
        eggs = []
        for key in all_eggs:
            for dist in all_eggs[key]:
                eggs.append(self.install_egg(dist.location, setup_base))
        if not eggs and (not self.dry_run):
            log.warn('No eggs found in %s (setup script problem?)', dist_dir)
        return eggs
    finally:
        _rmtree(dist_dir)
        log.set_verbosity(self.verbose)