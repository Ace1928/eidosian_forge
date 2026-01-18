from fontTools.misc.bezierTools import splitCubicAtTC
from collections import namedtuple
import math
from typing import (
@cython.locals(i=cython.int, j=cython.int, k=cython.int, start=cython.int, i_sol_count=cython.int, j_sol_count=cython.int, this_sol_count=cython.int, tolerance=cython.double, err=cython.double, error=cython.double, i_sol_error=cython.double, j_sol_error=cython.double, all_cubic=cython.int, is_cubic=cython.int, count=cython.int, p0=cython.complex, p1=cython.complex, p2=cython.complex, p3=cython.complex, v=cython.complex, u=cython.complex)
def spline_to_curves(q, costs, tolerance=0.5, all_cubic=False):
    """
    q: quadratic spline with alternating on-curve / off-curve points.

    costs: cumulative list of encoding cost of q in terms of number of
      points that need to be encoded.  Implied on-curve points do not
      contribute to the cost. If all points need to be encoded, then
      costs will be range(1, len(q)+1).
    """
    assert len(q) >= 3, 'quadratic spline requires at least 3 points'
    elevated_quadratics = [elevate_quadratic(*q[i:i + 3]) for i in range(0, len(q) - 2, 2)]
    forced = set()
    for i in range(1, len(elevated_quadratics)):
        p0 = elevated_quadratics[i - 1][2]
        p1 = elevated_quadratics[i][0]
        p2 = elevated_quadratics[i][1]
        if abs(p1 - p0) + abs(p2 - p1) > tolerance + abs(p2 - p0):
            forced.add(i)
    sols = [Solution(0, 0, 0, False)]
    impossible = Solution(len(elevated_quadratics) * 3 + 1, 0, 1, False)
    start = 0
    for i in range(1, len(elevated_quadratics) + 1):
        best_sol = impossible
        for j in range(start, i):
            j_sol_count, j_sol_error = (sols[j].num_points, sols[j].error)
            if not all_cubic:
                this_count = costs[2 * i - 1] - costs[2 * j] + 1
                i_sol_count = j_sol_count + this_count
                i_sol_error = j_sol_error
                i_sol = Solution(i_sol_count, i_sol_error, i - j, False)
                if i_sol < best_sol:
                    best_sol = i_sol
                if this_count <= 3:
                    continue
            try:
                curve, ts = merge_curves(elevated_quadratics, j, i - j)
            except ZeroDivisionError:
                continue
            reconstructed_iter = splitCubicAtTC(*curve, *ts)
            reconstructed = []
            error = 0
            for k, reconst in enumerate(reconstructed_iter):
                orig = elevated_quadratics[j + k]
                err = abs(reconst[3] - orig[3])
                error = max(error, err)
                if error > tolerance:
                    break
                reconstructed.append(reconst)
            if error > tolerance:
                continue
            for k, reconst in enumerate(reconstructed):
                orig = elevated_quadratics[j + k]
                p0, p1, p2, p3 = tuple((v - u for v, u in zip(reconst, orig)))
                if not cubic_farthest_fit_inside(p0, p1, p2, p3, tolerance):
                    error = tolerance + 1
                    break
            if error > tolerance:
                continue
            i_sol_count = j_sol_count + 3
            i_sol_error = max(j_sol_error, error)
            i_sol = Solution(i_sol_count, i_sol_error, i - j, True)
            if i_sol < best_sol:
                best_sol = i_sol
            if i_sol_count == 3:
                break
        sols.append(best_sol)
        if i in forced:
            start = i
    splits = []
    cubic = []
    i = len(sols) - 1
    while i:
        count, is_cubic = (sols[i].start_index, sols[i].is_cubic)
        splits.append(i)
        cubic.append(is_cubic)
        i -= count
    curves = []
    j = 0
    for i, is_cubic in reversed(list(zip(splits, cubic))):
        if is_cubic:
            curves.append(merge_curves(elevated_quadratics, j, i - j)[0])
        else:
            for k in range(j, i):
                curves.append(q[k * 2:k * 2 + 3])
        j = i
    return curves