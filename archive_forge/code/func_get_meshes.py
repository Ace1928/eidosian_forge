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
def get_meshes(self):
    np = import_module('numpy')
    mesh_x, mesh_y = np.meshgrid(np.linspace(self.start_x, self.end_x, num=self.nb_of_points_x), np.linspace(self.start_y, self.end_y, num=self.nb_of_points_y))
    f = vectorized_lambdify((self.var_x, self.var_y), self.expr)
    return (mesh_x, mesh_y, f(mesh_x, mesh_y))