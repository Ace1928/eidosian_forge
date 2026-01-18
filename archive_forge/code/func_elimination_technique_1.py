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
def elimination_technique_1(gens, rels, identity):
    rels = rels[:]
    rels.sort()
    gens = gens[:]
    redundant_gens = {}
    redundant_rels = []
    used_gens = set()
    for rel in rels:
        contained_gens = rel.contains_generators()
        if any((g in contained_gens for g in redundant_gens)):
            continue
        contained_gens = list(contained_gens)
        contained_gens.sort(reverse=True)
        for gen in contained_gens:
            if rel.generator_count(gen) == 1 and gen not in used_gens:
                k = rel.exponent_sum(gen)
                gen_index = rel.index(gen ** k)
                bk = rel.subword(gen_index + 1, len(rel))
                fw = rel.subword(0, gen_index)
                chi = bk * fw
                redundant_gens[gen] = chi ** (-1 * k)
                used_gens.update(chi.contains_generators())
                redundant_rels.append(rel)
                break
    rels = [r for r in rels if r not in redundant_rels]
    rels = [r.eliminate_words(redundant_gens, _all=True).identity_cyclic_reduction() for r in rels]
    rels = list(set(rels))
    try:
        rels.remove(identity)
    except ValueError:
        pass
    gens = [g for g in gens if g not in redundant_gens]
    return (gens, rels)