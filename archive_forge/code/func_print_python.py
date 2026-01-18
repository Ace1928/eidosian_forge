import keyword as kw
import sympy
from .repr import ReprPrinter
from .str import StrPrinter
def print_python(expr, **settings):
    """Print output of python() function"""
    print(python(expr, **settings))