from sympy.core import S, Symbol, diff, symbols
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import (Derivative, Function)
from sympy.core.mul import Mul
from sympy.core.relational import Eq
from sympy.core.sympify import sympify
from sympy.solvers import linsolve
from sympy.solvers.ode.ode import dsolve
from sympy.solvers.solvers import solve
from sympy.printing import sstr
from sympy.functions import SingularityFunction, Piecewise, factorial
from sympy.integrals import integrate
from sympy.series import limit
from sympy.plotting import plot, PlotGrid
from sympy.geometry.entity import GeometryEntity
from sympy.external import import_module
from sympy.sets.sets import Interval
from sympy.utilities.lambdify import lambdify
from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.iterables import iterable
def solve_for_ild_reactions(self, value, *reactions):
    """

        Determines the Influence Line Diagram equations for reaction
        forces under the effect of a moving load.

        Parameters
        ==========
        value : Integer
            Magnitude of moving load
        reactions :
            The reaction forces applied on the beam.

        Examples
        ========

        There is a beam of length 10 meters. There are two simple supports
        below the beam, one at the starting point and another at the ending
        point of the beam. Calculate the I.L.D. equations for reaction forces
        under the effect of a moving load of magnitude 1kN.

        Using the sign convention of downwards forces being positive.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy import symbols
            >>> from sympy.physics.continuum_mechanics.beam import Beam
            >>> E, I = symbols('E, I')
            >>> R_0, R_10 = symbols('R_0, R_10')
            >>> b = Beam(10, E, I)
            >>> b.apply_support(0, 'roller')
            >>> b.apply_support(10, 'roller')
            >>> b.solve_for_ild_reactions(1,R_0,R_10)
            >>> b.ild_reactions
            {R_0: x/10 - 1, R_10: -x/10}

        """
    shear_force, bending_moment = self._solve_for_ild_equations()
    x = self.variable
    l = self.length
    C3 = Symbol('C3')
    C4 = Symbol('C4')
    shear_curve = limit(shear_force, x, l) - value
    moment_curve = limit(bending_moment, x, l) - value * (l - x)
    slope_eqs = []
    deflection_eqs = []
    slope_curve = integrate(bending_moment, x) + C3
    for position, value in self._boundary_conditions['slope']:
        eqs = slope_curve.subs(x, position) - value
        slope_eqs.append(eqs)
    deflection_curve = integrate(slope_curve, x) + C4
    for position, value in self._boundary_conditions['deflection']:
        eqs = deflection_curve.subs(x, position) - value
        deflection_eqs.append(eqs)
    solution = list(linsolve([shear_curve, moment_curve] + slope_eqs + deflection_eqs, (C3, C4) + reactions).args[0])
    solution = solution[2:]
    self._ild_reactions = dict(zip(reactions, solution))