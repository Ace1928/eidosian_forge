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
def add_headers(self, *files):
    """Add installable headers to configuration.

        Add the given sequence of files to the beginning of the headers list.
        By default, headers will be installed under <python-
        include>/<self.name.replace('.','/')>/ directory. If an item of files
        is a tuple, then its first argument specifies the actual installation
        location relative to the <python-include> path.

        Parameters
        ----------
        files : str or seq
            Argument(s) can be either:

                * 2-sequence (<includedir suffix>,<path to header file(s)>)
                * path(s) to header file(s) where python includedir suffix will
                  default to package name.
        """
    headers = []
    for path in files:
        if is_string(path):
            [headers.append((self.name, p)) for p in self.paths(path)]
        else:
            if not isinstance(path, (tuple, list)) or len(path) != 2:
                raise TypeError(repr(path))
            [headers.append((path[0], p)) for p in self.paths(path[1])]
    dist = self.get_distribution()
    if dist is not None:
        if dist.headers is None:
            dist.headers = []
        dist.headers.extend(headers)
    else:
        self.headers.extend(headers)