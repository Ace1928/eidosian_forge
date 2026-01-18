from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_pp_directive(self, p):
    """ pp_directive  : PPHASH
        """
    self._parse_error('Directives not supported yet', self._token_coord(p, 1))