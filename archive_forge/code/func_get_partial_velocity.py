from sympy.core.backend import zeros, Matrix, diff, eye
from sympy.core.sorting import default_sort_key
from sympy.physics.vector import (ReferenceFrame, dynamicsymbols,
from sympy.physics.mechanics.method import _Methods
from sympy.physics.mechanics.particle import Particle
from sympy.physics.mechanics.rigidbody import RigidBody
from sympy.physics.mechanics.functions import (
from sympy.physics.mechanics.linearize import Linearizer
from sympy.utilities.iterables import iterable
def get_partial_velocity(body):
    if isinstance(body, RigidBody):
        vlist = [body.masscenter.vel(N), body.frame.ang_vel_in(N)]
    elif isinstance(body, Particle):
        vlist = [body.point.vel(N)]
    else:
        raise TypeError('The body list may only contain either RigidBody or Particle as list elements.')
    v = [msubs(vel, self._qdot_u_map) for vel in vlist]
    return partial_velocity(v, self.u, N)