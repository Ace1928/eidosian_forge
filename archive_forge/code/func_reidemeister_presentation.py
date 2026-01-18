from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.combinatorics.free_groups import (FreeGroup, FreeGroupElement,
from sympy.combinatorics.rewritingsystem import RewritingSystem
from sympy.combinatorics.coset_table import (CosetTable,
from sympy.combinatorics import PermutationGroup
from sympy.matrices.normalforms import invariant_factors
from sympy.matrices import Matrix
from sympy.polys.polytools import gcd
from sympy.printing.defaults import DefaultPrinting
from sympy.utilities import public
from sympy.utilities.magic import pollute
from itertools import product
def reidemeister_presentation(fp_grp, H, C=None, homomorphism=False):
    """
    Parameters
    ==========

    fp_group: A finitely presented group, an instance of FpGroup
    H: A subgroup whose presentation is to be found, given as a list
    of words in generators of `fp_grp`
    homomorphism: When set to True, return a homomorphism from the subgroup
                    to the parent group

    Examples
    ========

    >>> from sympy.combinatorics import free_group
    >>> from sympy.combinatorics.fp_groups import FpGroup, reidemeister_presentation
    >>> F, x, y = free_group("x, y")

    Example 5.6 Pg. 177 from [1]
    >>> f = FpGroup(F, [x**3, y**5, (x*y)**2])
    >>> H = [x*y, x**-1*y**-1*x*y*x]
    >>> reidemeister_presentation(f, H)
    ((y_1, y_2), (y_1**2, y_2**3, y_2*y_1*y_2*y_1*y_2*y_1))

    Example 5.8 Pg. 183 from [1]
    >>> f = FpGroup(F, [x**3, y**3, (x*y)**3])
    >>> H = [x*y, x*y**-1]
    >>> reidemeister_presentation(f, H)
    ((x_0, y_0), (x_0**3, y_0**3, x_0*y_0*x_0*y_0*x_0*y_0))

    Exercises Q2. Pg 187 from [1]
    >>> f = FpGroup(F, [x**2*y**2, y**-1*x*y*x**-3])
    >>> H = [x]
    >>> reidemeister_presentation(f, H)
    ((x_0,), (x_0**4,))

    Example 5.9 Pg. 183 from [1]
    >>> f = FpGroup(F, [x**3*y**-3, (x*y)**3, (x*y**-1)**2])
    >>> H = [x]
    >>> reidemeister_presentation(f, H)
    ((x_0,), (x_0**6,))

    """
    if not C:
        C = coset_enumeration_r(fp_grp, H)
    C.compress()
    C.standardize()
    define_schreier_generators(C, homomorphism=homomorphism)
    reidemeister_relators(C)
    gens, rels = (C._schreier_generators, C._reidemeister_relators)
    gens, rels = simplify_presentation(gens, rels, change_gens=True)
    C.schreier_generators = tuple(gens)
    C.reidemeister_relators = tuple(rels)
    if homomorphism:
        _gens = []
        for gen in gens:
            _gens.append(C._schreier_gen_elem[str(gen)])
        return (C.schreier_generators, C.reidemeister_relators, _gens)
    return (C.schreier_generators, C.reidemeister_relators)