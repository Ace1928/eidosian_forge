from importlib.metadata import version
import sympy
from sympy.external import import_module
from sympy.printing.str import StrPrinter
from sympy.physics.quantum.state import Bra, Ket
from .errors import LaTeXParsingError
def get_differential_var(d):
    text = get_differential_var_str(d.getText())
    return sympy.Symbol(text)