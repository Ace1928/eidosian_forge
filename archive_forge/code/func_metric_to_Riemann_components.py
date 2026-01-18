from __future__ import annotations
from typing import Any
from functools import reduce
from itertools import permutations
from sympy.combinatorics import Permutation
from sympy.core import (
from sympy.core.cache import cacheit
from sympy.core.symbol import Symbol, Dummy
from sympy.core.symbol import Str
from sympy.core.sympify import _sympify
from sympy.functions import factorial
from sympy.matrices import ImmutableDenseMatrix as Matrix
from sympy.solvers import solve
from sympy.utilities.exceptions import (sympy_deprecation_warning,
from sympy.tensor.array import ImmutableDenseNDimArray
from sympy.simplify.simplify import simplify
def metric_to_Riemann_components(expr):
    """Return the components of the Riemann tensor expressed in a given basis.

    Given a metric it calculates the components of the Riemann tensor in the
    canonical basis of the coordinate system in which the metric expression is
    given.

    Examples
    ========

    >>> from sympy import exp
    >>> from sympy.diffgeom.rn import R2
    >>> from sympy.diffgeom import metric_to_Riemann_components, TensorProduct
    >>> TP = TensorProduct

    >>> metric_to_Riemann_components(TP(R2.dx, R2.dx) + TP(R2.dy, R2.dy))
    [[[[0, 0], [0, 0]], [[0, 0], [0, 0]]], [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]]
    >>> non_trivial_metric = exp(2*R2.r)*TP(R2.dr, R2.dr) +         R2.r**2*TP(R2.dtheta, R2.dtheta)
    >>> non_trivial_metric
    exp(2*rho)*TensorProduct(drho, drho) + rho**2*TensorProduct(dtheta, dtheta)
    >>> riemann = metric_to_Riemann_components(non_trivial_metric)
    >>> riemann[0, :, :, :]
    [[[0, 0], [0, 0]], [[0, exp(-2*rho)*rho], [-exp(-2*rho)*rho, 0]]]
    >>> riemann[1, :, :, :]
    [[[0, -1/rho], [1/rho, 0]], [[0, 0], [0, 0]]]

    """
    ch_2nd = metric_to_Christoffel_2nd(expr)
    coord_sys = _find_coords(expr).pop()
    indices = list(range(coord_sys.dim))
    deriv_ch = [[[[d(ch_2nd[i, j, k]) for d in coord_sys.base_vectors()] for k in indices] for j in indices] for i in indices]
    riemann_a = [[[[deriv_ch[rho][sig][nu][mu] - deriv_ch[rho][sig][mu][nu] for nu in indices] for mu in indices] for sig in indices] for rho in indices]
    riemann_b = [[[[Add(*[ch_2nd[rho, l, mu] * ch_2nd[l, sig, nu] - ch_2nd[rho, l, nu] * ch_2nd[l, sig, mu] for l in indices]) for nu in indices] for mu in indices] for sig in indices] for rho in indices]
    riemann = [[[[riemann_a[rho][sig][mu][nu] + riemann_b[rho][sig][mu][nu] for nu in indices] for mu in indices] for sig in indices] for rho in indices]
    return ImmutableDenseNDimArray(riemann)