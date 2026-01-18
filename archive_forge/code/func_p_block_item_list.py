from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_block_item_list(self, p):
    """ block_item_list : block_item
                            | block_item_list block_item
        """
    p[0] = p[1] if len(p) == 2 or p[2] == [None] else p[1] + p[2]