from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_alignment_specifier(self, p):
    """ alignment_specifier  : _ALIGNAS LPAREN type_name RPAREN
                                 | _ALIGNAS LPAREN constant_expression RPAREN
        """
    p[0] = c_ast.Alignas(p[3], self._token_coord(p, 1))