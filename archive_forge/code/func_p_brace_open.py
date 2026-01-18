from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_brace_open(self, p):
    """ brace_open  :   LBRACE
        """
    p[0] = p[1]
    p.set_lineno(0, p.lineno(1))