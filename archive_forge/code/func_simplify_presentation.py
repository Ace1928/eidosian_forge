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
def simplify_presentation(*args, change_gens=False):
    """
    For an instance of `FpGroup`, return a simplified isomorphic copy of
    the group (e.g. remove redundant generators or relators). Alternatively,
    a list of generators and relators can be passed in which case the
    simplified lists will be returned.

    By default, the generators of the group are unchanged. If you would
    like to remove redundant generators, set the keyword argument
    `change_gens = True`.

    """
    if len(args) == 1:
        if not isinstance(args[0], FpGroup):
            raise TypeError('The argument must be an instance of FpGroup')
        G = args[0]
        gens, rels = simplify_presentation(G.generators, G.relators, change_gens=change_gens)
        if gens:
            return FpGroup(gens[0].group, rels)
        return FpGroup(FreeGroup([]), [])
    elif len(args) == 2:
        gens, rels = (args[0][:], args[1][:])
        if not gens:
            return (gens, rels)
        identity = gens[0].group.identity
    else:
        if len(args) == 0:
            m = 'Not enough arguments'
        else:
            m = 'Too many arguments'
        raise RuntimeError(m)
    prev_gens = []
    prev_rels = []
    while not set(prev_rels) == set(rels):
        prev_rels = rels
        while change_gens and (not set(prev_gens) == set(gens)):
            prev_gens = gens
            gens, rels = elimination_technique_1(gens, rels, identity)
        rels = _simplify_relators(rels, identity)
    if change_gens:
        syms = [g.array_form[0][0] for g in gens]
        F = free_group(syms)[0]
        identity = F.identity
        gens = F.generators
        subs = dict(zip(syms, gens))
        for j, r in enumerate(rels):
            a = r.array_form
            rel = identity
            for sym, p in a:
                rel = rel * subs[sym] ** p
            rels[j] = rel
    return (gens, rels)