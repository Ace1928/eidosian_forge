from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_abstract_declarator_3(self, p):
    """ abstract_declarator     : direct_abstract_declarator
        """
    p[0] = p[1]