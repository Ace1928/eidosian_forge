from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_jump_statement_3(self, p):
    """ jump_statement  : CONTINUE SEMI """
    p[0] = c_ast.Continue(self._token_coord(p, 1))