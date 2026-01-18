import re
from .ply import lex
from .ply.lex import TOKEN
@TOKEN(bad_char_const)
def t_BAD_CHAR_CONST(self, t):
    msg = 'Invalid char constant %s' % t.value
    self._error(msg, t)