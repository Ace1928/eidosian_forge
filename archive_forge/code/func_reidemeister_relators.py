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
def reidemeister_relators(C):
    R = C.fp_group.relators
    rels = [rewrite(C, coset, word) for word in R for coset in range(C.n)]
    order_1_gens = {i for i in rels if len(i) == 1}
    rels = list(filter(lambda rel: rel not in order_1_gens, rels))
    for i in range(len(rels)):
        w = rels[i]
        w = w.eliminate_words(order_1_gens, _all=True)
        rels[i] = w
    C._schreier_generators = [i for i in C._schreier_generators if not (i in order_1_gens or i ** (-1) in order_1_gens)]
    i = 0
    while i < len(rels):
        w = rels[i]
        j = i + 1
        while j < len(rels):
            if w.is_cyclic_conjugate(rels[j]):
                del rels[j]
            else:
                j += 1
        i += 1
    C._reidemeister_relators = rels