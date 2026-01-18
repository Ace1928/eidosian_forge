from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_struct_declarator_1(self, p):
    """ struct_declarator : declarator
        """
    p[0] = {'decl': p[1], 'bitsize': None}