from collections import namedtuple
from numba.core import types, ir
from numba.core.typing import signature
def make_tuple_variable(self, varlist, name='pf_tuple') -> ir.Var:
    """Makes a tuple variable

        Parameters
        ----------
        varlist : Sequence[ir.Var]
            Variables containing the values to be stored.
        name : str
            variable name to store to

        Returns
        -------
        res : ir.Var
        """
    loc = self._loc
    vartys = [self._typemap[x.name] for x in varlist]
    tupty = types.Tuple.from_types(vartys)
    return self.assign(rhs=ir.Expr.build_tuple(varlist, loc), typ=tupty, name=name)