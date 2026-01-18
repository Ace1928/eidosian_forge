import os
import atexit
import functools
import pickle
import sys
import time
import warnings
import numpy as np
def parprint(*args, **kwargs):
    """MPI-safe print - prints only from master. """
    if world.rank == 0:
        print(*args, **kwargs)