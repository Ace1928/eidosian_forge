from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_declaration_specifiers_no_type_4(self, p):
    """ declaration_specifiers_no_type  : atomic_specifier declaration_specifiers_no_type_opt
        """
    p[0] = self._add_declaration_specifier(p[2], p[1], 'type')