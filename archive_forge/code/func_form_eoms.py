from sympy.physics.mechanics import (Body, Lagrangian, KanesMethod, LagrangesMethod,
from sympy.physics.mechanics.method import _Methods
from sympy.core.backend import Matrix
def form_eoms(self, method=KanesMethod):
    """Method to form system's equation of motions.

        Parameters
        ==========

        method : Class
            Class name of method.

        Returns
        ========

        Matrix
            Vector of equations of motions.

        Examples
        ========

        This is a simple example for a one degree of freedom translational
        spring-mass-damper.

        >>> from sympy import S, symbols
        >>> from sympy.physics.mechanics import LagrangesMethod, dynamicsymbols, Body
        >>> from sympy.physics.mechanics import PrismaticJoint, JointsMethod
        >>> q = dynamicsymbols('q')
        >>> qd = dynamicsymbols('q', 1)
        >>> m, k, b = symbols('m k b')
        >>> wall = Body('W')
        >>> part = Body('P', mass=m)
        >>> part.potential_energy = k * q**2 / S(2)
        >>> J = PrismaticJoint('J', wall, part, coordinates=q, speeds=qd)
        >>> wall.apply_force(b * qd * wall.x, reaction_body=part)
        >>> method = JointsMethod(wall, J)
        >>> method.form_eoms(LagrangesMethod)
        Matrix([[b*Derivative(q(t), t) + k*q(t) + m*Derivative(q(t), (t, 2))]])

        We can also solve for the states using the 'rhs' method.

        >>> method.rhs()
        Matrix([
        [                Derivative(q(t), t)],
        [(-b*Derivative(q(t), t) - k*q(t))/m]])

        """
    bodylist = self._convert_bodies()
    if issubclass(method, LagrangesMethod):
        L = Lagrangian(self.frame, *bodylist)
        self._method = method(L, self.q, self.loads, bodylist, self.frame)
    else:
        self._method = method(self.frame, q_ind=self.q, u_ind=self.u, kd_eqs=self.kdes, forcelist=self.loads, bodies=bodylist)
    soln = self.method._form_eoms()
    return soln