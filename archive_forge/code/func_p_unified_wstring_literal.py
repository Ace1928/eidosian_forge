from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_unified_wstring_literal(self, p):
    """ unified_wstring_literal : WSTRING_LITERAL
                                    | U8STRING_LITERAL
                                    | U16STRING_LITERAL
                                    | U32STRING_LITERAL
                                    | unified_wstring_literal WSTRING_LITERAL
                                    | unified_wstring_literal U8STRING_LITERAL
                                    | unified_wstring_literal U16STRING_LITERAL
                                    | unified_wstring_literal U32STRING_LITERAL
        """
    if len(p) == 2:
        p[0] = c_ast.Constant('string', p[1], self._token_coord(p, 1))
    else:
        p[1].value = p[1].value.rstrip()[:-1] + p[2][2:]
        p[0] = p[1]