from __future__ import absolute_import, division, print_function
from . import lexer, error
from . import coretypes
def parse_datashape_nooption(self):
    """
        datashape_nooption : dim ASTERISK datashape
                           | dtype

        Returns a datashape object or None.
        """
    saved_pos = self.pos
    dim = self.parse_dim()
    if dim is not None:
        if self.tok.id == lexer.ASTERISK:
            self.advance_tok()
            saved_pos = self.pos
            dshape = self.parse_datashape()
            if dshape is None:
                self.pos = saved_pos
                self.raise_error('Expected a dim or a dtype')
            return coretypes.DataShape(dim, *dshape.parameters)
    dtype = self.parse_dtype()
    if dtype:
        return coretypes.DataShape(dtype)
    else:
        return None