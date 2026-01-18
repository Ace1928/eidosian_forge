from sympy.utilities import dict_merge
from sympy.utilities.iterables import iterable
from sympy.physics.vector import (Dyadic, Vector, ReferenceFrame,
from sympy.physics.vector.printing import (vprint, vsprint, vpprint, vlatex,
from sympy.physics.mechanics.particle import Particle
from sympy.physics.mechanics.rigidbody import RigidBody
from sympy.simplify.simplify import simplify
from sympy.core.backend import (Matrix, sympify, Mul, Derivative, sin, cos,
def inertia_of_point_mass(mass, pos_vec, frame):
    """Inertia dyadic of a point mass relative to point O.

    Parameters
    ==========

    mass : Sympifyable
        Mass of the point mass
    pos_vec : Vector
        Position from point O to point mass
    frame : ReferenceFrame
        Reference frame to express the dyadic in

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy.physics.mechanics import ReferenceFrame, inertia_of_point_mass
    >>> N = ReferenceFrame('N')
    >>> r, m = symbols('r m')
    >>> px = r * N.x
    >>> inertia_of_point_mass(m, px, N)
    m*r**2*(N.y|N.y) + m*r**2*(N.z|N.z)

    """
    return mass * (((frame.x | frame.x) + (frame.y | frame.y) + (frame.z | frame.z)) * (pos_vec & pos_vec) - (pos_vec | pos_vec))