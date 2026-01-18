from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_specifier_qualifier_list_1(self, p):
    """ specifier_qualifier_list    : specifier_qualifier_list type_specifier_no_typeid
        """
    p[0] = self._add_declaration_specifier(p[1], p[2], 'type', append=True)