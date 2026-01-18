from itertools import product, chain
from ..sage_helper import _within_sage
from ..math_basics import prod
from ..pari import pari
from .fundamental_polyhedron import *
def invariant_trace_field_generators(self):
    gens = self.generators()
    if min([abs(self(g).trace()) for g in gens]) < 0.001:
        raise ValueError('Algorithm fails when a generator has trace 0, see page 125 of ML')
    gens = [2 * g for g in gens]
    enough_elts = [''.join(sorted(s)) for s in powerset(gens) if len(s) > 0]
    return [self(w).trace() for w in enough_elts]