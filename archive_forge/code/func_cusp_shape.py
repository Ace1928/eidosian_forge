from itertools import product, chain
from ..sage_helper import _within_sage
from ..math_basics import prod
from ..pari import pari
from .fundamental_polyhedron import *
def cusp_shape(self, cusp_num=0):
    """
        Get the polished cusp shape for this representation::

          sage: M = ManifoldHP('m015')
          sage: rho = M.polished_holonomy(bits_prec=100)
          sage: rho.cusp_shape()   # doctest: +NUMERIC24
          -0.49024466750661447990098220731 + 2.9794470664789769463726817144*I

        """
    M, L = [self.SL2C(w) for w in self.peripheral_curves()[cusp_num]]
    C = extend_to_basis(parabolic_eigenvector(M))
    M, L = [make_trace_2(C ** (-1) * A * C) for A in [M, L]]
    z = L[0][1] / M[0][1]
    return z.conjugate()