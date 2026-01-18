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
def get_parameter_meshes(self):
    np = import_module('numpy')
    return np.meshgrid(np.linspace(self.start_u, self.end_u, num=self.nb_of_points_u), np.linspace(self.start_v, self.end_v, num=self.nb_of_points_v))