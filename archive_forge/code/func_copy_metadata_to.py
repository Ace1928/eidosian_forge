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
def copy_metadata_to(self, target_dir):
    """Copy metadata (egg info) to the target_dir"""
    norm_egg_info = os.path.normpath(self.egg_info)
    prefix = os.path.join(norm_egg_info, '')
    for path in self.ei_cmd.filelist.files:
        if path.startswith(prefix):
            target = os.path.join(target_dir, path[len(prefix):])
            ensure_directory(target)
            self.copy_file(path, target)