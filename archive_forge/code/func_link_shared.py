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
def link_shared(self, output, objects, libraries=(), library_dirs=(), export_symbols=(), extra_ldflags=None):
    """
        Create a shared library *output* linking the given *objects*
        and *libraries* (all strings).
        """
    output_dir, output_filename = os.path.split(output)
    self._compiler.link(CCompiler.SHARED_OBJECT, objects, output_filename, output_dir, libraries, library_dirs, export_symbols=export_symbols, extra_preargs=extra_ldflags)