from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
@parameterized(('id', 'ID'), ('typeid', 'TYPEID'))
def p_direct_xxx_declarator_2(self, p):
    """ direct_xxx_declarator   : LPAREN xxx_declarator RPAREN
        """
    p[0] = p[2]