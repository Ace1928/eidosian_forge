from sympy.core.backend import zeros, Matrix, diff, eye
from sympy.core.sorting import default_sort_key
from sympy.physics.vector import (ReferenceFrame, dynamicsymbols,
from sympy.physics.mechanics.method import _Methods
from sympy.physics.mechanics.particle import Particle
from sympy.physics.mechanics.rigidbody import RigidBody
from sympy.physics.mechanics.functions import (
from sympy.physics.mechanics.linearize import Linearizer
from sympy.utilities.iterables import iterable
@property
def mass_matrix_full(self):
    """The mass matrix of the system, augmented by the kinematic
        differential equations in explicit or implicit form."""
    if not self._fr or not self._frstar:
        raise ValueError('Need to compute Fr, Fr* first.')
    o, n = (len(self.u), len(self.q))
    return self.mass_matrix_kin.row_join(zeros(n, o)).col_join(zeros(o, n).row_join(self.mass_matrix))