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
def sorted_walk(dir):
    """Do os.walk in a reproducible way,
    independent of indeterministic filesystem readdir order
    """
    for base, dirs, files in os.walk(dir):
        dirs.sort()
        files.sort()
        yield (base, dirs, files)