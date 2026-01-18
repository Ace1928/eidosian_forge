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
def append_to(self, extlib):
    """Append libraries, include_dirs to extension or library item.
        """
    if is_sequence(extlib):
        lib_name, build_info = extlib
        dict_append(build_info, libraries=self.libraries, include_dirs=self.include_dirs)
    else:
        from numpy.distutils.core import Extension
        assert isinstance(extlib, Extension), repr(extlib)
        extlib.libraries.extend(self.libraries)
        extlib.include_dirs.extend(self.include_dirs)