import re
from .ply import lex
from .ply.lex import TOKEN
def t_pppragma_NEWLINE(self, t):
    """\\n"""
    t.lexer.lineno += 1
    t.lexer.begin('INITIAL')