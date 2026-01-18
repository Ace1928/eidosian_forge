import sys
import os
import shutil
import inspect
import tempfile
import subprocess
from contextlib import contextmanager
from functools import wraps
import numpy as np
from numpy.lib.recfunctions import repack_fields
import h5py
import unittest as ut
def mktemp_mpi(self, comm=None, suffix='.hdf5', prefix='', dir=None):
    if comm is None:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
    fname = None
    if comm.Get_rank() == 0:
        fname = self.mktemp(suffix, prefix, dir)
    fname = comm.bcast(fname, 0)
    return fname