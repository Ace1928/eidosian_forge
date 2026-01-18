from importlib.metadata import version
import sympy
from sympy.external import import_module
from sympy.printing.str import StrPrinter
from sympy.physics.quantum.state import Bra, Ket
from .errors import LaTeXParsingError
def handle_sum_or_prod(func, name):
    val = convert_mp(func.mp())
    iter_var = convert_expr(func.subeq().equality().expr(0))
    start = convert_expr(func.subeq().equality().expr(1))
    if func.supexpr().expr():
        end = convert_expr(func.supexpr().expr())
    else:
        end = convert_atom(func.supexpr().atom())
    if name == 'summation':
        return sympy.Sum(val, (iter_var, start, end))
    elif name == 'product':
        return sympy.Product(val, (iter_var, start, end))