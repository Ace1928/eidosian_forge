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
def mingw32():
    """Return true when using mingw32 environment.
    """
    if sys.platform == 'win32':
        if os.environ.get('OSTYPE', '') == 'msys':
            return True
        if os.environ.get('MSYSTEM', '') == 'MINGW32':
            return True
    return False