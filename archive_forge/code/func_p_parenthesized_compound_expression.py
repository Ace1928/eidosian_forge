from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_parenthesized_compound_expression(self, p):
    """ assignment_expression : LPAREN compound_statement RPAREN """
    p[0] = p[2]