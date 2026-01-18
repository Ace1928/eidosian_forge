from ctypes.util import find_library
from subprocess import check_output, CalledProcessError, DEVNULL
import ctypes
import os
import sys
import sysconfig
def get_libomp_names(self):
    """Return list of OpenMP libraries to try"""
    return ['omp', 'gomp', 'iomp5']