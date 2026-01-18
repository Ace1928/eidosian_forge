from sympy.core.numbers import (Float, I, oo, pi, zoo)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, asin, atan2, cos, sin, tan)
from sympy.matrices.dense import Matrix
from sympy.polys.polytools import cancel
from sympy.series.limits import Limit
from sympy.geometry.line import Ray3D
from sympy.geometry.util import intersection
from sympy.geometry.plane import Plane
from sympy.utilities.iterables import is_sequence
from .medium import Medium
def deviation(incident, medium1, medium2, normal=None, plane=None):
    """
    This function calculates the angle of deviation of a ray
    due to refraction at planar surface.

    Parameters
    ==========

    incident : Matrix, Ray3D, sequence or float
        Incident vector or angle of incidence
    medium1 : sympy.physics.optics.medium.Medium or sympifiable
        Medium 1 or its refractive index
    medium2 : sympy.physics.optics.medium.Medium or sympifiable
        Medium 2 or its refractive index
    normal : Matrix, Ray3D, or sequence
        Normal vector
    plane : Plane
        Plane of separation of the two media.

    Returns angular deviation between incident and refracted rays

    Examples
    ========

    >>> from sympy.physics.optics import deviation
    >>> from sympy.geometry import Point3D, Ray3D, Plane
    >>> from sympy.matrices import Matrix
    >>> from sympy import symbols
    >>> n1, n2 = symbols('n1, n2')
    >>> n = Matrix([0, 0, 1])
    >>> P = Plane(Point3D(0, 0, 0), normal_vector=[0, 0, 1])
    >>> r1 = Ray3D(Point3D(-1, -1, 1), Point3D(0, 0, 0))
    >>> deviation(r1, 1, 1, n)
    0
    >>> deviation(r1, n1, n2, plane=P)
    -acos(-sqrt(-2*n1**2/(3*n2**2) + 1)) + acos(-sqrt(3)/3)
    >>> round(deviation(0.1, 1.2, 1.5), 5)
    -0.02005
    """
    refracted = refraction_angle(incident, medium1, medium2, normal=normal, plane=plane)
    try:
        angle_of_incidence = Float(incident)
    except TypeError:
        angle_of_incidence = None
    if angle_of_incidence is not None:
        return float(refracted) - angle_of_incidence
    if refracted != 0:
        if isinstance(refracted, Ray3D):
            refracted = Matrix(refracted.direction_ratio)
        if not isinstance(incident, Matrix):
            if is_sequence(incident):
                _incident = Matrix(incident)
            elif isinstance(incident, Ray3D):
                _incident = Matrix(incident.direction_ratio)
            else:
                raise TypeError('incident should be a Matrix, Ray3D, or sequence')
        else:
            _incident = incident
        if plane is None:
            if not isinstance(normal, Matrix):
                if is_sequence(normal):
                    _normal = Matrix(normal)
                elif isinstance(normal, Ray3D):
                    _normal = Matrix(normal.direction_ratio)
                else:
                    raise TypeError('normal should be a Matrix, Ray3D, or sequence')
            else:
                _normal = normal
        else:
            _normal = Matrix(plane.normal_vector)
        mag_incident = sqrt(sum([i ** 2 for i in _incident]))
        mag_normal = sqrt(sum([i ** 2 for i in _normal]))
        mag_refracted = sqrt(sum([i ** 2 for i in refracted]))
        _incident /= mag_incident
        _normal /= mag_normal
        refracted /= mag_refracted
        i = acos(_incident.dot(_normal))
        r = acos(refracted.dot(_normal))
        return i - r