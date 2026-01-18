from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_constant_1(self, p):
    """ constant    : INT_CONST_DEC
                        | INT_CONST_OCT
                        | INT_CONST_HEX
                        | INT_CONST_BIN
                        | INT_CONST_CHAR
        """
    uCount = 0
    lCount = 0
    for x in p[1][-3:]:
        if x in ('l', 'L'):
            lCount += 1
        elif x in ('u', 'U'):
            uCount += 1
    t = ''
    if uCount > 1:
        raise ValueError('Constant cannot have more than one u/U suffix.')
    elif lCount > 2:
        raise ValueError('Constant cannot have more than two l/L suffix.')
    prefix = 'unsigned ' * uCount + 'long ' * lCount
    p[0] = c_ast.Constant(prefix + 'int', p[1], self._token_coord(p, 1))