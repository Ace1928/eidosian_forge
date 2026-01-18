import re
from .ply import lex
from .ply.lex import TOKEN
def reset_lineno(self):
    """ Resets the internal line number counter of the lexer.
        """
    self.lexer.lineno = 1