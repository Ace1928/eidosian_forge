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
def testNumericalSolutions():
    M = ManifoldGetter('m003')
    N = 3
    varieties = M.ptolemy_variety(N, obstruction_class='all')
    solutions = [solutions_from_magma(get_precomputed_magma(variety, dir=testing_files_generalized_directory), numerical=True) for variety in varieties]
    for obstruction_index, obstruction in enumerate(solutions):
        for component in obstruction:
            for solution in component:
                flattenings = solution.flattenings_numerical()
                if not test_regina:
                    flattenings.check_against_manifold(epsilon=1e-80)
                order = flattenings.get_order()
                if obstruction_index:
                    assert order == 6
                else:
                    assert order == 2
                cross_ratios = solution.cross_ratios()
                is_cr = cross_ratios.is_pu_2_1_representation(epsilon=1e-80, epsilon2=1e-10)
                if cross_ratios.volume_numerical().abs() < 1e-10:
                    assert is_cr
                else:
                    assert not is_cr
    number_one_dimensional = 0
    allComponents = sum(solutions, [])
    dimension_dict = {}
    degree_dict = {}
    for component in allComponents:
        dim = component.dimension
        deg = len(component)
        dimension_dict[dim] = 1 + dimension_dict.get(dim, 0)
        degree_dict[deg] = 1 + degree_dict.get(deg, 0)
        assert (dim == 0) ^ (deg == 0)
    assert dimension_dict == {0: 4, 1: 2}
    assert degree_dict == {0: 2, 2: 2, 8: 2}
    allSolutions = sum(allComponents, [])
    allCVolumes = [s.complex_volume_numerical() for s in allSolutions]
    expected_cvolume = pari('2.595387593686742138301993834077989475956329764530314161212797242812715071384508096863829303251915501 + 0.1020524924166561605528051801006522147774827678324290664524996742369032819581086580383974219370194645*I')
    expected_cvolumes = [pari(0), expected_cvolume, expected_cvolume.conj(), 2 * 4 * vol_tet]
    check_volumes(allCVolumes, expected_cvolumes)