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
def make_config_py(self, name='__config__'):
    """Generate package __config__.py file containing system_info
        information used during building the package.

        This file is installed to the
        package installation directory.

        """
    self.py_modules.append((self.name, name, generate_config_py))