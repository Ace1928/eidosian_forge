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
def apply_moment_load(self, value, start, order, dir='y'):
    """
        This method adds up the moment loads to a particular beam object.

        Parameters
        ==========
        value : Sympifyable
            The magnitude of an applied moment.
        dir : String
            Axis along which moment is applied.
        order : Integer
            The order of the applied load.
            - For point moments, order=-2
            - For constant distributed moment, order=-1
            - For ramp moments, order=0
            - For parabolic ramp moments, order=1
            - ... so on.
        """
    x = self.variable
    value = sympify(value)
    start = sympify(start)
    order = sympify(order)
    if dir == 'x':
        if not order == -2:
            self._moment_load_vector[0] += value
        elif start in list(self._torsion_moment):
            self._torsion_moment[start] += value
        else:
            self._torsion_moment[start] = value
        self._load_Singularity[0] += value * SingularityFunction(x, start, order)
    elif dir == 'y':
        if not order == -2:
            self._moment_load_vector[1] += value
        self._load_Singularity[0] += value * SingularityFunction(x, start, order)
    else:
        if not order == -2:
            self._moment_load_vector[2] += value
        self._load_Singularity[0] += value * SingularityFunction(x, start, order)