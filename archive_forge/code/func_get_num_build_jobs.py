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
def get_num_build_jobs():
    """
    Get number of parallel build jobs set by the --parallel command line
    argument of setup.py
    If the command did not receive a setting the environment variable
    NPY_NUM_BUILD_JOBS is checked. If that is unset, return the number of
    processors on the system, with a maximum of 8 (to prevent
    overloading the system if there a lot of CPUs).

    Returns
    -------
    out : int
        number of parallel jobs that can be run

    """
    from numpy.distutils.core import get_distribution
    try:
        cpu_count = len(os.sched_getaffinity(0))
    except AttributeError:
        cpu_count = multiprocessing.cpu_count()
    cpu_count = min(cpu_count, 8)
    envjobs = int(os.environ.get('NPY_NUM_BUILD_JOBS', cpu_count))
    dist = get_distribution()
    if dist is None:
        return envjobs
    cmdattr = (getattr(dist.get_command_obj('build'), 'parallel', None), getattr(dist.get_command_obj('build_ext'), 'parallel', None), getattr(dist.get_command_obj('build_clib'), 'parallel', None))
    if all((x is None for x in cmdattr)):
        return envjobs
    else:
        return max((x for x in cmdattr if x is not None))