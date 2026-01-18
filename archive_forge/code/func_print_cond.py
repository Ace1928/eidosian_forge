from sympy.core import S
from sympy.core.function import Lambda
from sympy.core.power import Pow
from .pycode import PythonCodePrinter, _known_functions_math, _print_known_const, _print_known_func, _unpack_integral_limits, ArrayPrinter
from .codeprinter import CodePrinter
def print_cond(cond):
    """ Problem having an ITE in the cond. """
    if cond.has(ITE):
        return self._print(simplify_logic(cond))
    else:
        return self._print(cond)