from importlib.metadata import version
import sympy
from sympy.external import import_module
from sympy.printing.str import StrPrinter
from sympy.physics.quantum.state import Bra, Ket
from .errors import LaTeXParsingError
def convert_atom(atom):
    if atom.LETTER():
        sname = atom.LETTER().getText()
        if atom.subexpr():
            if atom.subexpr().expr():
                subscript = convert_expr(atom.subexpr().expr())
            else:
                subscript = convert_atom(atom.subexpr().atom())
            sname += '_{' + StrPrinter().doprint(subscript) + '}'
        if atom.SINGLE_QUOTES():
            sname += atom.SINGLE_QUOTES().getText()
        return sympy.Symbol(sname)
    elif atom.SYMBOL():
        s = atom.SYMBOL().getText()[1:]
        if s == 'infty':
            return sympy.oo
        else:
            if atom.subexpr():
                subscript = None
                if atom.subexpr().expr():
                    subscript = convert_expr(atom.subexpr().expr())
                else:
                    subscript = convert_atom(atom.subexpr().atom())
                subscriptName = StrPrinter().doprint(subscript)
                s += '_{' + subscriptName + '}'
            return sympy.Symbol(s)
    elif atom.number():
        s = atom.number().getText().replace(',', '')
        return sympy.Number(s)
    elif atom.DIFFERENTIAL():
        var = get_differential_var(atom.DIFFERENTIAL())
        return sympy.Symbol('d' + var.name)
    elif atom.mathit():
        text = rule2text(atom.mathit().mathit_text())
        return sympy.Symbol(text)
    elif atom.frac():
        return convert_frac(atom.frac())
    elif atom.binom():
        return convert_binom(atom.binom())
    elif atom.bra():
        val = convert_expr(atom.bra().expr())
        return Bra(val)
    elif atom.ket():
        val = convert_expr(atom.ket().expr())
        return Ket(val)