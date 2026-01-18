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
def fresnel_coefficients(angle_of_incidence, medium1, medium2):
    """
    This function uses Fresnel equations to calculate reflection and
    transmission coefficients. Those are obtained for both polarisations
    when the electric field vector is in the plane of incidence (labelled 'p')
    and when the electric field vector is perpendicular to the plane of
    incidence (labelled 's'). There are four real coefficients unless the
    incident ray reflects in total internal in which case there are two complex
    ones. Angle of incidence is the angle between the incident ray and the
    surface normal. ``medium1`` and ``medium2`` can be ``Medium`` or any
    sympifiable object.

    Parameters
    ==========

    angle_of_incidence : sympifiable

    medium1 : Medium or sympifiable
        Medium 1 or its refractive index

    medium2 : Medium or sympifiable
        Medium 2 or its refractive index

    Returns
    =======

    Returns a list with four real Fresnel coefficients:
    [reflection p (TM), reflection s (TE),
    transmission p (TM), transmission s (TE)]
    If the ray is undergoes total internal reflection then returns a
    list of two complex Fresnel coefficients:
    [reflection p (TM), reflection s (TE)]

    Examples
    ========

    >>> from sympy.physics.optics import fresnel_coefficients
    >>> fresnel_coefficients(0.3, 1, 2)
    [0.317843553417859, -0.348645229818821,
            0.658921776708929, 0.651354770181179]
    >>> fresnel_coefficients(0.6, 2, 1)
    [-0.235625382192159 - 0.971843958291041*I,
             0.816477005968898 - 0.577377951366403*I]

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Fresnel_equations
    """
    if not 0 <= 2 * angle_of_incidence < pi:
        raise ValueError('Angle of incidence not in range [0:pi/2)')
    n1 = refractive_index_of_medium(medium1)
    n2 = refractive_index_of_medium(medium2)
    angle_of_refraction = asin(n1 * sin(angle_of_incidence) / n2)
    try:
        angle_of_total_internal_reflection_onset = critical_angle(n1, n2)
    except ValueError:
        angle_of_total_internal_reflection_onset = None
    if angle_of_total_internal_reflection_onset is None or angle_of_total_internal_reflection_onset > angle_of_incidence:
        R_s = -sin(angle_of_incidence - angle_of_refraction) / sin(angle_of_incidence + angle_of_refraction)
        R_p = tan(angle_of_incidence - angle_of_refraction) / tan(angle_of_incidence + angle_of_refraction)
        T_s = 2 * sin(angle_of_refraction) * cos(angle_of_incidence) / sin(angle_of_incidence + angle_of_refraction)
        T_p = 2 * sin(angle_of_refraction) * cos(angle_of_incidence) / (sin(angle_of_incidence + angle_of_refraction) * cos(angle_of_incidence - angle_of_refraction))
        return [R_p, R_s, T_p, T_s]
    else:
        n = n2 / n1
        R_s = cancel((cos(angle_of_incidence) - I * sqrt(sin(angle_of_incidence) ** 2 - n ** 2)) / (cos(angle_of_incidence) + I * sqrt(sin(angle_of_incidence) ** 2 - n ** 2)))
        R_p = cancel((n ** 2 * cos(angle_of_incidence) - I * sqrt(sin(angle_of_incidence) ** 2 - n ** 2)) / (n ** 2 * cos(angle_of_incidence) + I * sqrt(sin(angle_of_incidence) ** 2 - n ** 2)))
        return [R_p, R_s]