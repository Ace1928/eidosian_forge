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
def exec_mod_from_location(modname, modfile):
    """
    Use importlib machinery to import a module `modname` from the file
    `modfile`. Depending on the `spec.loader`, the module may not be
    registered in sys.modules.
    """
    spec = importlib.util.spec_from_file_location(modname, modfile)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    return foo