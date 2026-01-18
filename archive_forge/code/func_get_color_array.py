from collections.abc import Callable
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import arity, Function
from sympy.core.symbol import (Dummy, Symbol)
from sympy.core.sympify import sympify
from sympy.external import import_module
from sympy.printing.latex import latex
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import is_sequence
from .experimental_lambdify import (vectorized_lambdify, lambdify)
from sympy.plotting.textplot import textplot
def get_color_array(self):
    np = import_module('numpy')
    c = self.surface_color
    if isinstance(c, Callable):
        f = np.vectorize(c)
        nargs = arity(c)
        if self.is_parametric:
            variables = list(map(centers_of_faces, self.get_parameter_meshes()))
            if nargs == 1:
                return f(variables[0])
            elif nargs == 2:
                return f(*variables)
        variables = list(map(centers_of_faces, self.get_meshes()))
        if nargs == 1:
            return f(variables[0])
        elif nargs == 2:
            return f(*variables[:2])
        else:
            return f(*variables)
    elif isinstance(self, SurfaceOver2DRangeSeries):
        return c * np.ones(min(self.nb_of_points_x, self.nb_of_points_y))
    else:
        return c * np.ones(min(self.nb_of_points_u, self.nb_of_points_v))