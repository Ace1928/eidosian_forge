from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.polys.polytools import gcd
from sympy.sets.sets import Complement
from sympy.core import Basic, Tuple, diff, expand, Eq, Integer
from sympy.core.sorting import ordered
from sympy.core.symbol import _symbol
from sympy.solvers import solveset, nonlinsolve, diophantine
from sympy.polys import total_degree
from sympy.geometry import Point
from sympy.ntheory.factor_ import core
def rational_parametrization(self, parameters=('t', 's'), reg_point=None):
    """
        Returns the rational parametrization of implicit region.

        Examples
        ========

        >>> from sympy import Eq
        >>> from sympy.abc import x, y, z, s, t
        >>> from sympy.vector import ImplicitRegion

        >>> parabola = ImplicitRegion((x, y), y**2 - 4*x)
        >>> parabola.rational_parametrization()
        (4/t**2, 4/t)

        >>> circle = ImplicitRegion((x, y), Eq(x**2 + y**2, 4))
        >>> circle.rational_parametrization()
        (4*t/(t**2 + 1), 4*t**2/(t**2 + 1) - 2)

        >>> I = ImplicitRegion((x, y), x**3 + x**2 - y**2)
        >>> I.rational_parametrization()
        (t**2 - 1, t*(t**2 - 1))

        >>> cubic_curve = ImplicitRegion((x, y), x**3 + x**2 - y**2)
        >>> cubic_curve.rational_parametrization(parameters=(t))
        (t**2 - 1, t*(t**2 - 1))

        >>> sphere = ImplicitRegion((x, y, z), x**2 + y**2 + z**2 - 4)
        >>> sphere.rational_parametrization(parameters=(t, s))
        (-2 + 4/(s**2 + t**2 + 1), 4*s/(s**2 + t**2 + 1), 4*t/(s**2 + t**2 + 1))

        For some conics, regular_points() is unable to find a point on curve.
        To calulcate the parametric representation in such cases, user need
        to determine a point on the region and pass it using reg_point.

        >>> c = ImplicitRegion((x, y), (x  - 1/2)**2 + (y)**2 - (1/4)**2)
        >>> c.rational_parametrization(reg_point=(3/4, 0))
        (0.75 - 0.5/(t**2 + 1), -0.5*t/(t**2 + 1))

        References
        ==========

        - Christoph M. Hoffmann, "Conversion Methods between Parametric and
          Implicit Curves and Surfaces", Purdue e-Pubs, 1990. Available:
          https://docs.lib.purdue.edu/cgi/viewcontent.cgi?article=1827&context=cstech

        """
    equation = self.equation
    degree = self.degree
    if degree == 1:
        if len(self.variables) == 1:
            return (equation,)
        elif len(self.variables) == 2:
            x, y = self.variables
            y_par = list(solveset(equation, y))[0]
            return (x, y_par)
        else:
            raise NotImplementedError()
    point = ()
    if degree == 2:
        if reg_point is not None:
            point = reg_point
        elif len(self.singular_points()) != 0:
            point = list(self.singular_points())[0]
        else:
            point = self.regular_point()
    if len(self.singular_points()) != 0:
        singular_points = self.singular_points()
        for spoint in singular_points:
            syms = Tuple(*spoint).free_symbols
            rep = {s: 2 for s in syms}
            if len(syms) != 0:
                spoint = tuple((s.subs(rep) for s in spoint))
            if self.multiplicity(spoint) == degree - 1:
                point = spoint
                break
    if len(point) == 0:
        raise NotImplementedError()
    modified_eq = equation
    for i, var in enumerate(self.variables):
        modified_eq = modified_eq.subs(var, var + point[i])
    modified_eq = expand(modified_eq)
    hn = hn_1 = 0
    for term in modified_eq.args:
        if total_degree(term) == degree:
            hn += term
        else:
            hn_1 += term
    hn_1 = -1 * hn_1
    if not isinstance(parameters, tuple):
        parameters = (parameters,)
    if len(self.variables) == 2:
        parameter1 = parameters[0]
        if parameter1 == 's':
            s = _symbol('s_', real=True)
        else:
            s = _symbol('s', real=True)
        t = _symbol(parameter1, real=True)
        hn = hn.subs({self.variables[0]: s, self.variables[1]: t})
        hn_1 = hn_1.subs({self.variables[0]: s, self.variables[1]: t})
        x_par = (s * (hn_1 / hn)).subs(s, 1) + point[0]
        y_par = (t * (hn_1 / hn)).subs(s, 1) + point[1]
        return (x_par, y_par)
    elif len(self.variables) == 3:
        parameter1, parameter2 = parameters
        if 'r' in parameters:
            r = _symbol('r_', real=True)
        else:
            r = _symbol('r', real=True)
        s = _symbol(parameter2, real=True)
        t = _symbol(parameter1, real=True)
        hn = hn.subs({self.variables[0]: r, self.variables[1]: s, self.variables[2]: t})
        hn_1 = hn_1.subs({self.variables[0]: r, self.variables[1]: s, self.variables[2]: t})
        x_par = (r * (hn_1 / hn)).subs(r, 1) + point[0]
        y_par = (s * (hn_1 / hn)).subs(r, 1) + point[1]
        z_par = (t * (hn_1 / hn)).subs(r, 1) + point[2]
        return (x_par, y_par, z_par)
    raise NotImplementedError()