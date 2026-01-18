from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_static_assert_declaration(self, p):
    """ static_assert           : _STATIC_ASSERT LPAREN constant_expression COMMA unified_string_literal RPAREN
                                    | _STATIC_ASSERT LPAREN constant_expression RPAREN
        """
    if len(p) == 5:
        p[0] = [c_ast.StaticAssert(p[3], None, self._token_coord(p, 1))]
    else:
        p[0] = [c_ast.StaticAssert(p[3], p[5], self._token_coord(p, 1))]