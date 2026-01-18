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
def test_induced_sl4_representation():
    M = Manifold('m004')
    z_gl2 = ptolemy.CrossRatios.from_snappy_manifold(M)
    z_gl4 = z_gl2.induced_representation(4)
    G = M.fundamental_group()
    mat = z_gl4.evaluate_word(G.relators()[0], G)
    for i, row in enumerate(mat):
        for j, entry in enumerate(row):
            if i == j:
                assert abs(entry - 1) < 1e-09
            else:
                assert abs(entry) < 1e-09