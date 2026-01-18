from sympy.matrices.expressions.matexpr import MatrixExpr
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.functions.elementary.integers import floor
@property
def on_diag(self):
    return self.rowslice == self.colslice