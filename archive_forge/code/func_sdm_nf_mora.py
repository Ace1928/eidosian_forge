from itertools import permutations
from sympy.polys.monomials import (
from sympy.polys.polytools import Poly
from sympy.polys.polyutils import parallel_dict_from_expr
from sympy.core.singleton import S
from sympy.core.sympify import sympify
def sdm_nf_mora(f, G, O, K, phantom=None):
    """
    Compute a weak normal form of ``f`` with respect to ``G`` and order ``O``.

    The ground field is assumed to be ``K``, and monomials ordered according to
    ``O``.

    Weak normal forms are defined in [SCA, defn 2.3.3]. They are not unique.
    This function deterministically computes a weak normal form, depending on
    the order of `G`.

    The most important property of a weak normal form is the following: if
    `R` is the ring associated with the monomial ordering (if the ordering is
    global, we just have `R = K[x_1, \\ldots, x_n]`, otherwise it is a certain
    localization thereof), `I` any ideal of `R` and `G` a standard basis for
    `I`, then for any `f \\in R`, we have `f \\in I` if and only if
    `NF(f | G) = 0`.

    This is the generalized Mora algorithm for computing weak normal forms with
    respect to arbitrary monomial orders [SCA, algorithm 2.3.9].

    If ``phantom`` is not ``None``, it should be a pair of "phantom" arguments
    on which to perform the same computations as on ``f``, ``G``, both results
    are then returned.
    """
    from itertools import repeat
    h = f
    T = list(G)
    if phantom is not None:
        hp = phantom[0]
        Tp = list(phantom[1])
        phantom = True
    else:
        Tp = repeat([])
        phantom = False
    while h:
        Th = [(g, sdm_ecart(g), gp) for g, gp in zip(T, Tp) if sdm_monomial_divides(sdm_LM(g), sdm_LM(h))]
        if not Th:
            break
        g, _, gp = min(Th, key=lambda x: x[1])
        if sdm_ecart(g) > sdm_ecart(h):
            T.append(h)
            if phantom:
                Tp.append(hp)
        if phantom:
            h, hp = sdm_spoly(h, g, O, K, phantom=(hp, gp))
        else:
            h = sdm_spoly(h, g, O, K)
    if phantom:
        return (h, hp)
    return h