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
def get_python_libraries(self):
    """
        Get the library arguments necessary to link with Python.
        """
    libs = self._build_ext.get_libraries(_DummyExtension())
    if sys.platform == 'win32':
        libs = libs + ['msvcrt']
    return libs + self._math_info['libraries']