from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_postfix_expression_2(self, p):
    """ postfix_expression  : postfix_expression LBRACKET expression RBRACKET """
    p[0] = c_ast.ArrayRef(p[1], p[3], p[1].coord)