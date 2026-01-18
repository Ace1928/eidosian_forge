import os
import atexit
import functools
import pickle
import sys
import time
import warnings
import numpy as np
def paropen(name, mode='r', buffering=-1, encoding=None, comm=None):
    """MPI-safe version of open function.

    In read mode, the file is opened on all nodes.  In write and
    append mode, the file is opened on the master only, and /dev/null
    is opened on all other nodes.
    """
    if comm is None:
        comm = world
    if comm.rank > 0 and mode[0] != 'r':
        name = os.devnull
    return open(name, mode, buffering, encoding)