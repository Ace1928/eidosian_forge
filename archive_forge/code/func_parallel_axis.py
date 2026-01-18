from sympy.core.backend import sympify
from sympy.physics.vector import Point, ReferenceFrame, Dyadic
from sympy.utilities.exceptions import sympy_deprecation_warning
def parallel_axis(self, point, frame=None):
    """Returns the inertia dyadic of the body with respect to another
        point.

        Parameters
        ==========

        point : sympy.physics.vector.Point
            The point to express the inertia dyadic about.
        frame : sympy.physics.vector.ReferenceFrame
            The reference frame used to construct the dyadic.

        Returns
        =======

        inertia : sympy.physics.vector.Dyadic
            The inertia dyadic of the rigid body expressed about the provided
            point.

        """
    from sympy.physics.mechanics.functions import inertia_of_point_mass
    if frame is None:
        frame = self.frame
    return self.central_inertia + inertia_of_point_mass(self.mass, self.masscenter.pos_from(point), frame)