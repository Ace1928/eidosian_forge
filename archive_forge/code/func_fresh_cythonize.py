import difflib
import glob
import gzip
import os
import sys
import tempfile
import unittest
import Cython.Build.Dependencies
import Cython.Utils
from Cython.TestUtils import CythonTest
def fresh_cythonize(self, *args, **kwargs):
    Cython.Utils.clear_function_caches()
    Cython.Build.Dependencies._dep_tree = None
    Cython.Build.Dependencies.cythonize(*args, **kwargs)