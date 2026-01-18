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
def has_f_sources(sources):
    """Return True if sources contains Fortran files """
    for source in sources:
        if fortran_ext_match(source):
            return True
    return False