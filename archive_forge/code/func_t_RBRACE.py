import re
from .ply import lex
from .ply.lex import TOKEN
@TOKEN('\\}')
def t_RBRACE(self, t):
    self.on_rbrace_func()
    return t