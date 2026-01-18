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
def assertSameElements(self, a, b):
    for x in a:
        match = False
        for y in b:
            if x == y:
                match = True
        if not match:
            raise AssertionError("Item '%s' appears in a but not b" % x)
    for x in b:
        match = False
        for y in a:
            if x == y:
                match = True
        if not match:
            raise AssertionError("Item '%s' appears in b but not a" % x)