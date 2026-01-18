from __future__ import absolute_import, division, print_function
from . import lexer, error
from . import coretypes
def parse_struct_field(self):
    """
        struct_field : struct_field_name COLON datashape
        struct_field_name : NAME_LOWER
                          | NAME_UPPER
                          | NAME_OTHER
                          | STRING

        Returns a tuple (name, datashape object) or None
        """
    if self.tok.id not in [lexer.NAME_LOWER, lexer.NAME_UPPER, lexer.NAME_OTHER, lexer.STRING]:
        return None
    name = self.tok.val
    self.advance_tok()
    if self.tok.id != lexer.COLON:
        self.raise_error('Expected a ":" separating the field ' + 'name from its datashape')
    self.advance_tok()
    ds = self.parse_datashape()
    if ds is None:
        self.raise_error('Expected the datashape of the field')
    return (name, ds)