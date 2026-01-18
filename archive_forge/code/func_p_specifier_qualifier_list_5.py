from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_specifier_qualifier_list_5(self, p):
    """ specifier_qualifier_list  : alignment_specifier
        """
    p[0] = dict(qual=[], alignment=[p[1]], storage=[], type=[], function=[])