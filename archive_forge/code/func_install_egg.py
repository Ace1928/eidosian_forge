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
def install_egg(self, egg_path, tmpdir):
    destination = os.path.join(self.install_dir, os.path.basename(egg_path))
    destination = os.path.abspath(destination)
    if not self.dry_run:
        ensure_directory(destination)
    dist = self.egg_distribution(egg_path)
    if not (os.path.exists(destination) and os.path.samefile(egg_path, destination)):
        if os.path.isdir(destination) and (not os.path.islink(destination)):
            dir_util.remove_tree(destination, dry_run=self.dry_run)
        elif os.path.exists(destination):
            self.execute(os.unlink, (destination,), 'Removing ' + destination)
        try:
            new_dist_is_zipped = False
            if os.path.isdir(egg_path):
                if egg_path.startswith(tmpdir):
                    f, m = (shutil.move, 'Moving')
                else:
                    f, m = (shutil.copytree, 'Copying')
            elif self.should_unzip(dist):
                self.mkpath(destination)
                f, m = (self.unpack_and_compile, 'Extracting')
            else:
                new_dist_is_zipped = True
                if egg_path.startswith(tmpdir):
                    f, m = (shutil.move, 'Moving')
                else:
                    f, m = (shutil.copy2, 'Copying')
            self.execute(f, (egg_path, destination), (m + ' %s to %s') % (os.path.basename(egg_path), os.path.dirname(destination)))
            update_dist_caches(destination, fix_zipimporter_caches=new_dist_is_zipped)
        except Exception:
            update_dist_caches(destination, fix_zipimporter_caches=False)
            raise
    self.add_output(destination)
    return self.egg_distribution(destination)