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
def checkSolutionsForManifoldGeneralizedObstructionClass(solutions_trivial, solutions_non_trivial, manifold, N, baseline_volumes, baseline_dimensions):
    torsionTrivial = pari('Pi^2/6 * I')
    torsionNonTrivial = pari('Pi^2/18 * I')
    solutions = [(s, False) for s in solutions_trivial] + [(s, True) for s in solutions_non_trivial]
    dimensions = set()
    volumes = []
    volumes_2 = []
    for solution, sol_is_non_trivial in solutions:
        dimensions.add(solution.dimension)
        if solution.dimension == 0:
            if sol_is_non_trivial:
                got_exception = False
                try:
                    solution.check_against_manifold(manifold)
                except PtolemyCannotBeCheckedError:
                    got_exception = True
                    assert got_exception, 'check_against_manifold should not have passed'
            else:
                solution.check_against_manifold(manifold)
            fl = solution.flattenings_numerical()
            for f in fl:
                if not test_regina:
                    f.check_against_manifold(epsilon=1e-80)
                cvol, modulo = f.complex_volume(with_modulo=True)
                if sol_is_non_trivial and N == 3:
                    assert (modulo - torsionNonTrivial).abs() < 1e-80, 'Wrong modulo returned non-trivial case'
                else:
                    assert (modulo - torsionTrivial).abs() < 1e-80, 'Wrong modulo returned trivial case'
                volumes_2.append(cvol.real())
            volumes += solution.volume_numerical()
            cross_ratios = solution.cross_ratios()
            if not test_regina:
                cross_ratios.check_against_manifold(manifold)

    def is_close(a, b):
        return (a - b).abs() < 1e-80

    def make_unique(L):
        L.sort()
        result = L[:1]
        for i in L:
            if not is_close(result[-1], i):
                result.append(i)
        return result
    volumes = make_unique(volumes)
    volumes_2 = make_unique(volumes_2)
    all_expected_volumes = make_unique(baseline_volumes + [-vol for vol in baseline_volumes])
    assert len(all_expected_volumes) >= 2 * len(baseline_volumes) - 1
    for volume, expected_volume in zip(volumes, all_expected_volumes):
        assert is_close(volume, expected_volume)
    for volume, expected_volume in zip(volumes_2, all_expected_volumes):
        assert is_close(volume, expected_volume)
    assert dimensions == set(baseline_dimensions)