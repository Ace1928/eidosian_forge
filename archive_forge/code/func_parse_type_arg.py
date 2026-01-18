from __future__ import absolute_import, division, print_function
from . import lexer, error
from . import coretypes
def parse_type_arg(self):
    """
        type_arg : datashape
                 | INTEGER
                 | STRING
                 | BOOLEAN
                 | list_type_arg
        list_type_arg : LBRACKET RBRACKET
                      | LBRACKET datashape_list RBRACKET
                      | LBRACKET integer_list RBRACKET
                      | LBRACKET string_list RBRACKET

        Returns a type_arg value, or None.
        """
    ds = self.parse_datashape()
    if ds is not None:
        return ds
    if self.tok.id in [lexer.INTEGER, lexer.STRING, lexer.BOOLEAN]:
        val = self.tok.val
        self.advance_tok()
        return val
    elif self.tok.id == lexer.LBRACKET:
        self.advance_tok()
        val = self.parse_datashape_list()
        if val is None:
            val = self.parse_integer_list()
        if val is None:
            val = self.parse_string_list()
        if val is None:
            val = self.parse_boolean_list()
        if self.tok.id == lexer.RBRACKET:
            self.advance_tok()
            return [] if val is None else val
        elif val is None:
            self.raise_error('Expected a type constructor argument ' + 'or a closing "]"')
        else:
            self.raise_error('Expected a "," or a closing "]"')
    else:
        return None