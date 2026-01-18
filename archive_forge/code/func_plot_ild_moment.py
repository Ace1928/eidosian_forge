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
def plot_ild_moment(self, subs=None):
    """

        Plots the Influence Line Diagram for Moment under the effect
        of a moving load. This function should be called after
        calling solve_for_ild_moment().

        Parameters
        ==========

        subs : dictionary
               Python dictionary containing Symbols as key and their
               corresponding values.

        Examples
        ========

        There is a beam of length 12 meters. There are two simple supports
        below the beam, one at the starting point and another at a distance
        of 8 meters. Plot the I.L.D. for Moment at a distance
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
            >>> b.solve_for_ild_moment(4, 1, R_0, R_8)
            >>> b.ild_moment
            Piecewise((-x/2, x < 4), (x/2 - 4, x > 4))
            >>> b.plot_ild_moment()
            Plot object containing:
            [0]: cartesian line: Piecewise((-x/2, x < 4), (x/2 - 4, x > 4)) for x over (0.0, 12.0)

        """
    if not self._ild_moment:
        raise ValueError('I.L.D. moment equation not found. Please use solve_for_ild_moment() to generate the I.L.D. moment equations.')
    x = self.variable
    if subs is None:
        subs = {}
    for sym in self._ild_moment.atoms(Symbol):
        if sym != x and sym not in subs:
            raise ValueError('Value of %s was not passed.' % sym)
    for sym in self._length.atoms(Symbol):
        if sym != x and sym not in subs:
            raise ValueError('Value of %s was not passed.' % sym)
    return plot(self._ild_moment.subs(subs), (x, 0, self._length), title='I.L.D. for Moment', xlabel='$\\mathrm{X}$', ylabel='$\\mathrm{M}$', line_color='blue', show=True)