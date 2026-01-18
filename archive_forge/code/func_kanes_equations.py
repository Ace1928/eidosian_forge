from sympy.core.backend import zeros, Matrix, diff, eye
from sympy.core.sorting import default_sort_key
from sympy.physics.vector import (ReferenceFrame, dynamicsymbols,
from sympy.physics.mechanics.method import _Methods
from sympy.physics.mechanics.particle import Particle
from sympy.physics.mechanics.rigidbody import RigidBody
from sympy.physics.mechanics.functions import (
from sympy.physics.mechanics.linearize import Linearizer
from sympy.utilities.iterables import iterable
def kanes_equations(self, bodies=None, loads=None):
    """ Method to form Kane's equations, Fr + Fr* = 0.

        Explanation
        ===========

        Returns (Fr, Fr*). In the case where auxiliary generalized speeds are
        present (say, s auxiliary speeds, o generalized speeds, and m motion
        constraints) the length of the returned vectors will be o - m + s in
        length. The first o - m equations will be the constrained Kane's
        equations, then the s auxiliary Kane's equations. These auxiliary
        equations can be accessed with the auxiliary_eqs property.

        Parameters
        ==========

        bodies : iterable
            An iterable of all RigidBody's and Particle's in the system.
            A system must have at least one body.
        loads : iterable
            Takes in an iterable of (Particle, Vector) or (ReferenceFrame, Vector)
            tuples which represent the force at a point or torque on a frame.
            Must be either a non-empty iterable of tuples or None which corresponds
            to a system with no constraints.
        """
    if bodies is None:
        bodies = self.bodies
    if loads is None and self._forcelist is not None:
        loads = self._forcelist
    if loads == []:
        loads = None
    if not self._k_kqdot:
        raise AttributeError('Create an instance of KanesMethod with kinematic differential equations to use this method.')
    fr = self._form_fr(loads)
    frstar = self._form_frstar(bodies)
    if self._uaux:
        if not self._udep:
            km = KanesMethod(self._inertial, self.q, self._uaux, u_auxiliary=self._uaux)
        else:
            km = KanesMethod(self._inertial, self.q, self._uaux, u_auxiliary=self._uaux, u_dependent=self._udep, velocity_constraints=self._k_nh * self.u + self._f_nh, acceleration_constraints=self._k_dnh * self._udot + self._f_dnh)
        km._qdot_u_map = self._qdot_u_map
        self._km = km
        fraux = km._form_fr(loads)
        frstaraux = km._form_frstar(bodies)
        self._aux_eq = fraux + frstaraux
        self._fr = fr.col_join(fraux)
        self._frstar = frstar.col_join(frstaraux)
    return (self._fr, self._frstar)