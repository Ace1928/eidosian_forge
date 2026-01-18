from itertools import product, chain
from ..sage_helper import _within_sage
from ..math_basics import prod
from ..pari import pari
from .fundamental_polyhedron import *
def all_lifts_to_SL2C(self):
    ans = []
    self.lift_to_SL2C()
    base_gen_images = [self(g) for g in self.generators()]
    pos_signs = product(*[(1, -1)] * len(base_gen_images))
    for signs in pos_signs:
        beta = ManifoldGroup(self.generators(), self.relators(), self.peripheral_curves(), [s * A for s, A in zip(signs, base_gen_images)])
        if beta.is_nonprojective_representation():
            ans.append(beta)
    return ans