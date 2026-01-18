import os
import re
import sys
import copy
import glob
import atexit
import tempfile
import subprocess
import shutil
import multiprocessing
import textwrap
import importlib.util
from threading import local as tlocal
from functools import reduce
import distutils
from distutils.errors import DistutilsError
def add_include_dirs(self, *paths):
    """Add paths to configuration include directories.

        Add the given sequence of paths to the beginning of the include_dirs
        list. This list will be visible to all extension modules of the
        current package.
        """
    include_dirs = self.paths(paths)
    dist = self.get_distribution()
    if dist is not None:
        if dist.include_dirs is None:
            dist.include_dirs = []
        dist.include_dirs.extend(include_dirs)
    else:
        self.include_dirs.extend(include_dirs)