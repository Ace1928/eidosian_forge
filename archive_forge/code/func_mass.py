from sympy.core.backend import sympify
from sympy.physics.vector import Point, ReferenceFrame, Dyadic
from sympy.utilities.exceptions import sympy_deprecation_warning
@mass.setter
def mass(self, m):
    self._mass = sympify(m)