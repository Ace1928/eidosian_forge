import setuptools
from setuptools.command.build_ext import build_ext
from setuptools.dist import Distribution
import numpy as np
import functools
import os
import subprocess
import sys
from tempfile import mkdtemp
from contextlib import contextmanager
from pathlib import Path
def get_python_include_dirs(self):
    """
        Get the include directories necessary to compile against the Python
        and Numpy C APIs.
        """
    return list(self._py_include_dirs) + self._math_info['include_dirs']