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
def get_numpy_include_dirs():
    include_dirs = Configuration.numpy_include_dirs[:]
    if not include_dirs:
        import numpy
        include_dirs = [numpy.get_include()]
    return include_dirs