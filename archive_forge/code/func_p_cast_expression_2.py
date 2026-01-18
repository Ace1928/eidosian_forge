from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_cast_expression_2(self, p):
    """ cast_expression : LPAREN type_name RPAREN cast_expression """
    p[0] = c_ast.Cast(p[2], p[4], self._token_coord(p, 1))