from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_external_declaration_2(self, p):
    """ external_declaration    : declaration
        """
    p[0] = p[1]