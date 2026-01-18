from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
@parameterized(('id', 'ID'), ('typeid', 'TYPEID'), ('typeid_noparen', 'TYPEID'))
def p_direct_xxx_declarator_1(self, p):
    """ direct_xxx_declarator   : yyy
        """
    p[0] = c_ast.TypeDecl(declname=p[1], type=None, quals=None, align=None, coord=self._token_coord(p, 1))