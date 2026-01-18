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
def gpaths(paths, local_path='', include_non_existing=True):
    """Apply glob to paths and prepend local_path if needed.
    """
    if is_string(paths):
        paths = (paths,)
    return _fix_paths(paths, local_path, include_non_existing)