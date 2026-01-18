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
def max_shear_force(self):
    """
        Returns point of max shear force and its corresponding shear value
        along all directions in a Beam object as a list.
        solve_for_reaction_loads() must be called before using this function.

        Examples
        ========
        There is a beam of length 20 meters. It it supported by rollers
        at of its end. A linear load having slope equal to 12 is applied
        along y-axis. A constant distributed load of magnitude 15 N is
        applied from start till its end along z-axis.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam3D
            >>> from sympy import symbols
            >>> l, E, G, I, A, x = symbols('l, E, G, I, A, x')
            >>> b = Beam3D(20, 40, 21, 100, 25, x)
            >>> b.apply_load(15, start=0, order=0, dir="z")
            >>> b.apply_load(12*x, start=0, order=0, dir="y")
            >>> b.bc_deflection = [(0, [0, 0, 0]), (20, [0, 0, 0])]
            >>> R1, R2, R3, R4 = symbols('R1, R2, R3, R4')
            >>> b.apply_load(R1, start=0, order=-1, dir="z")
            >>> b.apply_load(R2, start=20, order=-1, dir="z")
            >>> b.apply_load(R3, start=0, order=-1, dir="y")
            >>> b.apply_load(R4, start=20, order=-1, dir="y")
            >>> b.solve_for_reaction_loads(R1, R2, R3, R4)
            >>> b.max_shear_force()
            [(0, 0), (20, 2400), (20, 300)]
        """
    max_shear = []
    max_shear.append(self._max_shear_force('x'))
    max_shear.append(self._max_shear_force('y'))
    max_shear.append(self._max_shear_force('z'))
    return max_shear