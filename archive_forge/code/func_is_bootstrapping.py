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
def is_bootstrapping():
    import builtins
    try:
        builtins.__NUMPY_SETUP__
        return True
    except AttributeError:
        return False