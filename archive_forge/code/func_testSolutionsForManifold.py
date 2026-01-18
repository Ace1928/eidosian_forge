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
def testSolutionsForManifold(M, N, solutions, baseline_cvolumes=None, expect_non_zero_dimensional=None, against_geometric=True):
    old_precision = pari.set_real_precision(100)
    found_non_zero_dimensional = False
    numerical_solutions = []
    numerical_cross_ratios = []
    numerical_cross_ratios_alt = []
    for solution in solutions:
        if isinstance(solution, ptolemy.component.NonZeroDimensionalComponent):
            found_non_zero_dimensional = True
        else:
            assert solution.N() == N
            assert solution.num_tetrahedra() == M.num_tetrahedra()
            solution.check_against_manifold(M)
            for numerical_solution in solution.numerical():
                numerical_solutions.append(numerical_solution)
                numerical_cross_ratios.append(numerical_solution.cross_ratios())
            cross_ratios = solution.cross_ratios()
            if not test_regina:
                cross_ratios.check_against_manifold(M)
            assert cross_ratios.N() == N
            numerical_cross_ratios_alt += cross_ratios.numerical()
    if expect_non_zero_dimensional is not None:
        assert expect_non_zero_dimensional == found_non_zero_dimensional
    for s in numerical_solutions:
        s.check_against_manifold(M, epsilon=1e-80)
        if not test_regina:
            s.flattenings_numerical().check_against_manifold(M, epsilon=1e-80)
    if not test_regina:
        for s in numerical_cross_ratios:
            s.check_against_manifold(M, epsilon=1e-80)
        for s in numerical_cross_ratios_alt:
            s.check_against_manifold(M, epsilon=1e-80)
    complex_volumes = [s.complex_volume_numerical() for s in numerical_solutions]
    volumes = [s.volume_numerical() for s in numerical_cross_ratios]
    assert len(complex_volumes) == len(volumes)
    for vol, cvol in zip(volumes, complex_volumes):
        diff = vol - cvol.real()
        assert diff.abs() < 1e-80
    volumes_alt = [s.volume_numerical() for s in numerical_cross_ratios_alt]
    assert len(volumes) == len(volumes_alt)
    volumes.sort(key=float)
    volumes_alt.sort(key=float)
    for vol1, vol2 in zip(volumes, volumes_alt):
        assert (vol1 - vol2).abs() < 1e-80
    if against_geometric and (not test_regina):
        if M.solution_type() == 'all tetrahedra positively oriented':
            geom_vol = M.volume() * (N - 1) * N * (N + 1) / 6
            assert True in [abs(geom_vol - vol) < 1e-11 for vol in volumes]
    if baseline_cvolumes is not None:
        check_volumes(complex_volumes, baseline_cvolumes)
    pari.set_real_precision(old_precision)