from __future__ import absolute_import, division, print_function
from . import lexer, error
from . import coretypes
def syntactic_sugar(self, symdict, name, dshapemsg, error_pos=None):
    """
        Looks up a symbol in the provided symbol table dictionary for
        syntactic sugar, raising a standard error message if the symbol
        is missing.

        Parameters
        ----------
        symdict : symbol table dictionary
            One of self.sym.dtype, self.sym.dim,
            self.sym.dtype_constr, or self.sym.dim_constr.
        name : str
            The name of the symbol to look up.
        dshapemsg : str
            The datashape construct this lookup is for, e.g.
            '{...} dtype constructor'.
        error_pos : int, optional
            The position in the token stream at which to flag the error.
        """
    entry = symdict.get(name)
    if entry is not None:
        return entry
    else:
        if error_pos is not None:
            self.pos = error_pos
        self.raise_error(('Symbol table missing "%s" ' + 'entry for %s') % (name, dshapemsg))