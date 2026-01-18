from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_jump_statement_1(self, p):
    """ jump_statement  : GOTO ID SEMI """
    p[0] = c_ast.Goto(p[2], self._token_coord(p, 1))