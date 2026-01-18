from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_init_declarator_list(self, p):
    """ init_declarator_list    : init_declarator
                                    | init_declarator_list COMMA init_declarator
        """
    p[0] = p[1] + [p[3]] if len(p) == 4 else [p[1]]