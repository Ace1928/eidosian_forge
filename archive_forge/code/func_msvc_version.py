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
def msvc_version(compiler):
    """Return version major and minor of compiler instance if it is
    MSVC, raise an exception otherwise."""
    if not compiler.compiler_type == 'msvc':
        raise ValueError('Compiler instance is not msvc (%s)' % compiler.compiler_type)
    return compiler._MSVCCompiler__version