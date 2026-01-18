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
def solve_for_ild_shear(self, distance, value, *reactions):
    """

        Determines the Influence Line Diagram equations for shear at a
        specified point under the effect of a moving load.

        Parameters
        ==========
        distance : Integer
            Distance of the point from the start of the beam
            for which equations are to be determined
        value : Integer
            Magnitude of moving load
        reactions :
            The reaction forces applied on the beam.

        Examples
        ========

        There is a beam of length 12 meters. There are two simple supports
        below the beam, one at the starting point and another at a distance
        of 8 meters. Calculate the I.L.D. equations for Shear at a distance
        of 4 meters under the effect of a moving load of magnitude 1kN.

        Using the sign convention of downwards forces being positive.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy import symbols
            >>> from sympy.physics.continuum_mechanics.beam import Beam
            >>> E, I = symbols('E, I')
            >>> R_0, R_8 = symbols('R_0, R_8')
            >>> b = Beam(12, E, I)
            >>> b.apply_support(0, 'roller')
            >>> b.apply_support(8, 'roller')
            >>> b.solve_for_ild_reactions(1, R_0, R_8)
            >>> b.solve_for_ild_shear(4, 1, R_0, R_8)
            >>> b.ild_shear
            Piecewise((x/8, x < 4), (x/8 - 1, x > 4))

        """
    x = self.variable
    l = self.length
    shear_force, _ = self._solve_for_ild_equations()
    shear_curve1 = value - limit(shear_force, x, distance)
    shear_curve2 = limit(shear_force, x, l) - limit(shear_force, x, distance) - value
    for reaction in reactions:
        shear_curve1 = shear_curve1.subs(reaction, self._ild_reactions[reaction])
        shear_curve2 = shear_curve2.subs(reaction, self._ild_reactions[reaction])
    shear_eq = Piecewise((shear_curve1, x < distance), (shear_curve2, x > distance))
    self._ild_shear = shear_eq