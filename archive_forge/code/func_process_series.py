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
def process_series(self):
    """
        Iterates over every ``Plot`` object and further calls
        _process_series()
        """
    parent = self.parent
    if isinstance(parent, Plot):
        series_list = [parent._series]
    else:
        series_list = parent._series
    for i, (series, ax) in enumerate(zip(series_list, self.ax)):
        if isinstance(self.parent, PlotGrid):
            parent = self.parent.args[i]
        self._process_series(series, ax, parent)