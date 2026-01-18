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
def msvc_runtime_major():
    """Return major version of MSVC runtime coded like get_build_msvc_version"""
    major = {1300: 70, 1310: 71, 1400: 80, 1500: 90, 1600: 100, 1900: 140}.get(msvc_runtime_version(), None)
    return major