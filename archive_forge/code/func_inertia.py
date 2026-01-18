from sympy.core.backend import sympify
from sympy.physics.vector import Point, ReferenceFrame, Dyadic
from sympy.utilities.exceptions import sympy_deprecation_warning
@inertia.setter
def inertia(self, I):
    if not isinstance(I[0], Dyadic):
        raise TypeError('RigidBody inertia must be a Dyadic object.')
    if not isinstance(I[1], Point):
        raise TypeError('RigidBody inertia must be about a Point.')
    self._inertia = I[0]
    self._inertia_point = I[1]
    from sympy.physics.mechanics.functions import inertia_of_point_mass
    I_Ss_O = inertia_of_point_mass(self.mass, self.masscenter.pos_from(I[1]), self.frame)
    self._central_inertia = I[0] - I_Ss_O