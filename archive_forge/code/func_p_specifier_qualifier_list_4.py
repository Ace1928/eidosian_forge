from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_specifier_qualifier_list_4(self, p):
    """ specifier_qualifier_list  : type_qualifier_list type_specifier
        """
    p[0] = dict(qual=p[1], alignment=[], storage=[], type=[p[2]], function=[])