from sympy.core.backend import sympify
from sympy.physics.vector import Point, ReferenceFrame, Dyadic
from sympy.utilities.exceptions import sympy_deprecation_warning
def set_potential_energy(self, scalar):
    sympy_deprecation_warning('\nThe sympy.physics.mechanics.RigidBody.set_potential_energy()\nmethod is deprecated. Instead use\n\n    B.potential_energy = scalar\n            ', deprecated_since_version='1.5', active_deprecations_target='deprecated-set-potential-energy')
    self.potential_energy = scalar