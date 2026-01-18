from importlib.metadata import version
import sympy
from sympy.external import import_module
from sympy.printing.str import StrPrinter
from sympy.physics.quantum.state import Bra, Ket
from .errors import LaTeXParsingError
def syntaxError(self, recog, symbol, line, col, msg, e):
    fmt = '%s\n%s\n%s'
    marker = '~' * col + '^'
    if msg.startswith('missing'):
        err = fmt % (msg, self.src, marker)
    elif msg.startswith('no viable'):
        err = fmt % ('I expected something else here', self.src, marker)
    elif msg.startswith('mismatched'):
        names = LaTeXParser.literalNames
        expected = [names[i] for i in e.getExpectedTokens() if i < len(names)]
        if len(expected) < 10:
            expected = ' '.join(expected)
            err = fmt % ('I expected one of these: ' + expected, self.src, marker)
        else:
            err = fmt % ('I expected something else here', self.src, marker)
    else:
        err = fmt % ("I don't understand this", self.src, marker)
    raise LaTeXParsingError(err)