from snappy import Manifold, pari, ptolemy
from snappy.ptolemy import solutions_from_magma, Flattenings, parse_solutions
from snappy.ptolemy.processFileBase import get_manifold
from snappy.ptolemy import __path__ as ptolemy_paths
from snappy.ptolemy.coordinates import PtolemyCannotBeCheckedError
from snappy.sage_helper import _within_sage, doctest_modules
from snappy.pari import pari
import bz2
import os
import sys
def matrix_is_diagonal(m):
    return m[0][0] - m[1][1] == 0 and m[0][1] == 0 and (m[1][0] == 0)