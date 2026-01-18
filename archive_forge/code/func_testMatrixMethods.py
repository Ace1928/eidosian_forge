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
def testMatrixMethods(manifold, solutions):

    def matrix_is_diagonal(m):
        return m[0][0] - m[1][1] == 0 and m[0][1] == 0 and (m[1][0] == 0)

    def matrix_is_pm_identity(m):
        return matrix_is_diagonal(m) and (m[0][0] + 1 == 0 or m[0][0] - 1 == 0)
    print('Testing matrix methods...')
    G = manifold.fundamental_group(simplify_presentation=True)
    Graw = manifold.fundamental_group(simplify_presentation=False)
    for solution in solutions:
        if solution.dimension == 0:
            solution._testing_check_cocycles()
            cross_ratios = solution.cross_ratios()
            for gen in G.generators():
                assert not matrix_is_diagonal(solution.evaluate_word(gen, G))
                assert not matrix_is_diagonal(cross_ratios.evaluate_word(gen, G))
            for gen in Graw.generators():
                assert not matrix_is_diagonal(solution.evaluate_word(gen, Graw))
                assert not matrix_is_diagonal(cross_ratios.evaluate_word(gen, Graw))
                assert not matrix_is_diagonal(solution.evaluate_word(gen))
                assert not matrix_is_diagonal(cross_ratios.evaluate_word(gen))
            for rel in G.relators():
                assert matrix_is_pm_identity(solution.evaluate_word(rel, G))
                assert matrix_is_diagonal(solution.evaluate_word(rel, G))
            for rel in Graw.relators():
                assert matrix_is_pm_identity(solution.evaluate_word(rel, Graw))
                assert matrix_is_diagonal(solution.evaluate_word(rel, Graw))
                assert matrix_is_pm_identity(solution.evaluate_word(rel))
                assert matrix_is_diagonal(solution.evaluate_word(rel))