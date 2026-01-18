from importlib.metadata import version
import sympy
from sympy.external import import_module
from sympy.printing.str import StrPrinter
from sympy.physics.quantum.state import Bra, Ket
from .errors import LaTeXParsingError
def convert_add(add):
    if add.ADD():
        lh = convert_add(add.additive(0))
        rh = convert_add(add.additive(1))
        return sympy.Add(lh, rh, evaluate=False)
    elif add.SUB():
        lh = convert_add(add.additive(0))
        rh = convert_add(add.additive(1))
        if hasattr(rh, 'is_Atom') and rh.is_Atom:
            return sympy.Add(lh, -1 * rh, evaluate=False)
        return sympy.Add(lh, sympy.Mul(-1, rh, evaluate=False), evaluate=False)
    else:
        return convert_mp(add.mp())