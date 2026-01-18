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
def install_item(self, spec, download, tmpdir, deps, install_needed=False):
    install_needed = install_needed or self.always_copy
    install_needed = install_needed or os.path.dirname(download) == tmpdir
    install_needed = install_needed or not download.endswith('.egg')
    install_needed = install_needed or (self.always_copy_from is not None and os.path.dirname(normalize_path(download)) == normalize_path(self.always_copy_from))
    if spec and (not install_needed):
        for dist in self.local_index[spec.project_name]:
            if dist.location == download:
                break
        else:
            install_needed = True
    log.info('Processing %s', os.path.basename(download))
    if install_needed:
        dists = self.install_eggs(spec, download, tmpdir)
        for dist in dists:
            self.process_distribution(spec, dist, deps)
    else:
        dists = [self.egg_distribution(download)]
        self.process_distribution(spec, dists[0], deps, 'Using')
    if spec is not None:
        for dist in dists:
            if dist in spec:
                return dist
    return None