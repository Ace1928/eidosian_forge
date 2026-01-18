from __future__ import absolute_import, division, print_function
from . import lexer, error
from . import coretypes
def parse_funcproto_or_tuple_type(self):
    """
        funcproto_or_tuple_type : tuple_type RARROW datashape
                                | tuple_type
        tuple_type : LPAREN tuple_item_list RPAREN
                   | LPAREN tuple_item_list COMMA RPAREN
                   | LPAREN RPAREN
        tuple_item_list : datashape COMMA tuple_item_list
                        | datashape

        Returns a tuple type object, a function prototype, or None.
        """
    if self.tok.id != lexer.LPAREN:
        return None
    saved_pos = self.pos
    self.advance_tok()
    dshapes = self.parse_homogeneous_list(self.parse_datashape, lexer.COMMA, 'Invalid datashape in tuple', trailing_sep=True) or ()
    if self.tok.id != lexer.RPAREN:
        self.raise_error('Invalid datashape in tuple')
    self.advance_tok()
    if self.tok.id != lexer.RARROW:
        tconstr = self.syntactic_sugar(self.sym.dtype_constr, 'tuple', '(...) dtype constructor', saved_pos)
        return tconstr(dshapes)
    else:
        self.advance_tok()
        ret_dshape = self.parse_datashape()
        if ret_dshape is None:
            self.raise_error('Expected function prototype return ' + 'datashape')
        tconstr = self.syntactic_sugar(self.sym.dtype_constr, 'funcproto', '(...) -> ... dtype constructor', saved_pos)
        return tconstr(dshapes, ret_dshape)