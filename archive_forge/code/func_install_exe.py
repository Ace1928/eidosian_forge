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
def install_exe(self, dist_filename, tmpdir):
    cfg = extract_wininst_cfg(dist_filename)
    if cfg is None:
        raise DistutilsError('%s is not a valid distutils Windows .exe' % dist_filename)
    dist = Distribution(None, project_name=cfg.get('metadata', 'name'), version=cfg.get('metadata', 'version'), platform=get_platform())
    egg_path = os.path.join(tmpdir, dist.egg_name() + '.egg')
    dist.location = egg_path
    egg_tmp = egg_path + '.tmp'
    _egg_info = os.path.join(egg_tmp, 'EGG-INFO')
    pkg_inf = os.path.join(_egg_info, 'PKG-INFO')
    ensure_directory(pkg_inf)
    dist._provider = PathMetadata(egg_tmp, _egg_info)
    self.exe_to_egg(dist_filename, egg_tmp)
    if not os.path.exists(pkg_inf):
        f = open(pkg_inf, 'w')
        f.write('Metadata-Version: 1.0\n')
        for k, v in cfg.items('metadata'):
            if k != 'target_version':
                f.write('%s: %s\n' % (k.replace('_', '-').title(), v))
        f.close()
    script_dir = os.path.join(_egg_info, 'scripts')
    self.delete_blockers([os.path.join(script_dir, args[0]) for args in ScriptWriter.get_args(dist)])
    bdist_egg.make_zipfile(egg_path, egg_tmp, verbose=self.verbose, dry_run=self.dry_run)
    return self.install_egg(egg_path, tmpdir)