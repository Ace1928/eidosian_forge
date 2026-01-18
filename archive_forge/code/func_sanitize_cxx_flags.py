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
def sanitize_cxx_flags(cxxflags):
    """
    Some flags are valid for C but not C++. Prune them.
    """
    return [flag for flag in cxxflags if flag not in _cxx_ignore_flags]