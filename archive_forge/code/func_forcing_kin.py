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
def forcing_kin(self):
    """The kinematic "forcing vector" of the system."""
    if self.explicit_kinematics:
        return -(self._k_ku * Matrix(self.u) + self._f_k)
    else:
        return -(self._k_ku_implicit * Matrix(self.u) + self._f_k_implicit)