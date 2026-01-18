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
def refraction_angle(incident, medium1, medium2, normal=None, plane=None):
    """
    This function calculates transmitted vector after refraction at planar
    surface. ``medium1`` and ``medium2`` can be ``Medium`` or any sympifiable object.
    If ``incident`` is a number then treated as angle of incidence (in radians)
    in which case refraction angle is returned.

    If ``incident`` is an object of `Ray3D`, `normal` also has to be an instance
    of `Ray3D` in order to get the output as a `Ray3D`. Please note that if
    plane of separation is not provided and normal is an instance of `Ray3D`,
    ``normal`` will be assumed to be intersecting incident ray at the plane of
    separation. This will not be the case when `normal` is a `Matrix` or
    any other sequence.
    If ``incident`` is an instance of `Ray3D` and `plane` has not been provided
    and ``normal`` is not `Ray3D`, output will be a `Matrix`.

    Parameters
    ==========

    incident : Matrix, Ray3D, sequence or a number
        Incident vector or angle of incidence
    medium1 : sympy.physics.optics.medium.Medium or sympifiable
        Medium 1 or its refractive index
    medium2 : sympy.physics.optics.medium.Medium or sympifiable
        Medium 2 or its refractive index
    normal : Matrix, Ray3D, or sequence
        Normal vector
    plane : Plane
        Plane of separation of the two media.

    Returns
    =======

    Returns an angle of refraction or a refracted ray depending on inputs.

    Examples
    ========

    >>> from sympy.physics.optics import refraction_angle
    >>> from sympy.geometry import Point3D, Ray3D, Plane
    >>> from sympy.matrices import Matrix
    >>> from sympy import symbols, pi
    >>> n = Matrix([0, 0, 1])
    >>> P = Plane(Point3D(0, 0, 0), normal_vector=[0, 0, 1])
    >>> r1 = Ray3D(Point3D(-1, -1, 1), Point3D(0, 0, 0))
    >>> refraction_angle(r1, 1, 1, n)
    Matrix([
    [ 1],
    [ 1],
    [-1]])
    >>> refraction_angle(r1, 1, 1, plane=P)
    Ray3D(Point3D(0, 0, 0), Point3D(1, 1, -1))

    With different index of refraction of the two media

    >>> n1, n2 = symbols('n1, n2')
    >>> refraction_angle(r1, n1, n2, n)
    Matrix([
    [                                n1/n2],
    [                                n1/n2],
    [-sqrt(3)*sqrt(-2*n1**2/(3*n2**2) + 1)]])
    >>> refraction_angle(r1, n1, n2, plane=P)
    Ray3D(Point3D(0, 0, 0), Point3D(n1/n2, n1/n2, -sqrt(3)*sqrt(-2*n1**2/(3*n2**2) + 1)))
    >>> round(refraction_angle(pi/6, 1.2, 1.5), 5)
    0.41152
    """
    n1 = refractive_index_of_medium(medium1)
    n2 = refractive_index_of_medium(medium2)
    try:
        angle_of_incidence = float(incident)
    except TypeError:
        angle_of_incidence = None
    try:
        critical_angle_ = critical_angle(medium1, medium2)
    except (ValueError, TypeError):
        critical_angle_ = None
    if angle_of_incidence is not None:
        if normal is not None or plane is not None:
            raise ValueError('Normal/plane not allowed if incident is an angle')
        if not 0.0 <= angle_of_incidence < pi * 0.5:
            raise ValueError('Angle of incidence not in range [0:pi/2)')
        if critical_angle_ and angle_of_incidence > critical_angle_:
            raise ValueError('Ray undergoes total internal reflection')
        return asin(n1 * sin(angle_of_incidence) / n2)
    return_ray = False
    if plane is not None and normal is not None:
        raise ValueError('Either plane or normal is acceptable.')
    if not isinstance(incident, Matrix):
        if is_sequence(incident):
            _incident = Matrix(incident)
        elif isinstance(incident, Ray3D):
            _incident = Matrix(incident.direction_ratio)
        else:
            raise TypeError('incident should be a Matrix, Ray3D, or sequence')
    else:
        _incident = incident
    if plane is not None:
        if not isinstance(plane, Plane):
            raise TypeError('plane should be an instance of geometry.plane.Plane')
        if isinstance(incident, Ray3D):
            return_ray = True
            intersection_pt = plane.intersection(incident)[0]
        _normal = Matrix(plane.normal_vector)
    elif not isinstance(normal, Matrix):
        if is_sequence(normal):
            _normal = Matrix(normal)
        elif isinstance(normal, Ray3D):
            _normal = Matrix(normal.direction_ratio)
            if isinstance(incident, Ray3D):
                intersection_pt = intersection(incident, normal)
                if len(intersection_pt) == 0:
                    raise ValueError("Normal isn't concurrent with the incident ray.")
                else:
                    return_ray = True
                    intersection_pt = intersection_pt[0]
        else:
            raise TypeError('Normal should be a Matrix, Ray3D, or sequence')
    else:
        _normal = normal
    eta = n1 / n2
    mag_incident = sqrt(sum([i ** 2 for i in _incident]))
    mag_normal = sqrt(sum([i ** 2 for i in _normal]))
    _incident /= mag_incident
    _normal /= mag_normal
    c1 = -_incident.dot(_normal)
    cs2 = 1 - eta ** 2 * (1 - c1 ** 2)
    if cs2.is_negative:
        return S.Zero
    drs = eta * _incident + (eta * c1 - sqrt(cs2)) * _normal
    drs = drs * mag_incident
    if not return_ray:
        return drs
    else:
        return Ray3D(intersection_pt, direction_ratio=drs)