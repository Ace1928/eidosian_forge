from importlib.metadata import version
import sympy
from sympy.external import import_module
from sympy.printing.str import StrPrinter
from sympy.physics.quantum.state import Bra, Ket
from .errors import LaTeXParsingError
def parse_latex(sympy):
    antlr4 = import_module('antlr4')
    if None in [antlr4, MathErrorListener] or not version('antlr4-python3-runtime').startswith('4.11'):
        raise ImportError('LaTeX parsing requires the antlr4 Python package, provided by pip (antlr4-python3-runtime) or conda (antlr-python-runtime), version 4.11')
    matherror = MathErrorListener(sympy)
    stream = antlr4.InputStream(sympy)
    lex = LaTeXLexer(stream)
    lex.removeErrorListeners()
    lex.addErrorListener(matherror)
    tokens = antlr4.CommonTokenStream(lex)
    parser = LaTeXParser(tokens)
    parser.removeErrorListeners()
    parser.addErrorListener(matherror)
    relation = parser.math().relation()
    expr = convert_relation(relation)
    return expr