from collections import defaultdict
from itertools import product
from functools import reduce
from math import prod
from sympy import SYMPY_DEBUG
from sympy.core import (S, Dummy, symbols, sympify, Tuple, expand, I, pi, Mul,
from sympy.core.mod import Mod
from sympy.core.sorting import default_sort_key
from sympy.functions import (exp, sqrt, root, log, lowergamma, cos,
from sympy.functions.elementary.complexes import polarify, unpolarify
from sympy.functions.special.hyper import (hyper, HyperRep_atanh,
from sympy.matrices import Matrix, eye, zeros
from sympy.polys import apart, poly, Poly
from sympy.series import residue
from sympy.simplify.powsimp import powdenest
from sympy.utilities.iterables import sift
def devise_plan_meijer(fro, to, z):
    """
    Find operators to convert G-function ``fro`` into G-function ``to``.

    Explanation
    ===========

    It is assumed that ``fro`` and ``to`` have the same signatures, and that in fact
    any corresponding pair of parameters differs by integers, and a direct path
    is possible. I.e. if there are parameters a1 b1 c1  and a2 b2 c2 it is
    assumed that a1 can be shifted to a2, etc. The only thing this routine
    determines is the order of shifts to apply, nothing clever will be tried.
    It is also assumed that ``fro`` is suitable.

    Examples
    ========

    >>> from sympy.simplify.hyperexpand import (devise_plan_meijer,
    ...                                         G_Function)
    >>> from sympy.abc import z

    Empty plan:

    >>> devise_plan_meijer(G_Function([1], [2], [3], [4]),
    ...                    G_Function([1], [2], [3], [4]), z)
    []

    Very simple plans:

    >>> devise_plan_meijer(G_Function([0], [], [], []),
    ...                    G_Function([1], [], [], []), z)
    [<Increment upper a index #0 of [0], [], [], [].>]
    >>> devise_plan_meijer(G_Function([0], [], [], []),
    ...                    G_Function([-1], [], [], []), z)
    [<Decrement upper a=0.>]
    >>> devise_plan_meijer(G_Function([], [1], [], []),
    ...                    G_Function([], [2], [], []), z)
    [<Increment lower a index #0 of [], [1], [], [].>]

    Slightly more complicated plans:

    >>> devise_plan_meijer(G_Function([0], [], [], []),
    ...                    G_Function([2], [], [], []), z)
    [<Increment upper a index #0 of [1], [], [], [].>,
    <Increment upper a index #0 of [0], [], [], [].>]
    >>> devise_plan_meijer(G_Function([0], [], [0], []),
    ...                    G_Function([-1], [], [1], []), z)
    [<Increment upper b=0.>, <Decrement upper a=0.>]

    Order matters:

    >>> devise_plan_meijer(G_Function([0], [], [0], []),
    ...                    G_Function([1], [], [1], []), z)
    [<Increment upper a index #0 of [0], [], [1], [].>, <Increment upper b=0.>]
    """

    def try_shift(f, t, shifter, diff, counter):
        """ Try to apply ``shifter`` in order to bring some element in ``f``
            nearer to its counterpart in ``to``. ``diff`` is +/- 1 and
            determines the effect of ``shifter``. Counter is a list of elements
            blocking the shift.

            Return an operator if change was possible, else None.
        """
        for idx, (a, b) in enumerate(zip(f, t)):
            if (a - b).is_integer and (b - a) / diff > 0 and all((a != x for x in counter)):
                sh = shifter(idx)
                f[idx] += diff
                return sh
    fan = list(fro.an)
    fap = list(fro.ap)
    fbm = list(fro.bm)
    fbq = list(fro.bq)
    ops = []
    change = True
    while change:
        change = False
        op = try_shift(fan, to.an, lambda i: MeijerUnShiftB(fan, fap, fbm, fbq, i, z), 1, fbm + fbq)
        if op is not None:
            ops += [op]
            change = True
            continue
        op = try_shift(fap, to.ap, lambda i: MeijerUnShiftD(fan, fap, fbm, fbq, i, z), 1, fbm + fbq)
        if op is not None:
            ops += [op]
            change = True
            continue
        op = try_shift(fbm, to.bm, lambda i: MeijerUnShiftA(fan, fap, fbm, fbq, i, z), -1, fan + fap)
        if op is not None:
            ops += [op]
            change = True
            continue
        op = try_shift(fbq, to.bq, lambda i: MeijerUnShiftC(fan, fap, fbm, fbq, i, z), -1, fan + fap)
        if op is not None:
            ops += [op]
            change = True
            continue
        op = try_shift(fan, to.an, lambda i: MeijerShiftB(fan[i]), -1, [])
        if op is not None:
            ops += [op]
            change = True
            continue
        op = try_shift(fap, to.ap, lambda i: MeijerShiftD(fap[i]), -1, [])
        if op is not None:
            ops += [op]
            change = True
            continue
        op = try_shift(fbm, to.bm, lambda i: MeijerShiftA(fbm[i]), 1, [])
        if op is not None:
            ops += [op]
            change = True
            continue
        op = try_shift(fbq, to.bq, lambda i: MeijerShiftC(fbq[i]), 1, [])
        if op is not None:
            ops += [op]
            change = True
            continue
    if fan != list(to.an) or fap != list(to.ap) or fbm != list(to.bm) or (fbq != list(to.bq)):
        raise NotImplementedError('Could not devise plan.')
    ops.reverse()
    return ops