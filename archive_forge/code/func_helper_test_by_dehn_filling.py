import re
import string
def helper_test_by_dehn_filling(M):
    from snappy import Manifold
    M_filled = M.filled_triangulation()
    for ignore_cusp_ordering in [False, True]:
        for ignore_curve_orientations in [False, True]:
            isosig = M.triangulation_isosig(decorated=True, ignore_cusp_ordering=ignore_cusp_ordering, ignore_curve_orientations=ignore_curve_orientations)
            N = Manifold(isosig)
            N_filled = N.filled_triangulation()
            helper_are_isometric(M, N)