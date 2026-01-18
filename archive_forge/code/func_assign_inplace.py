from collections import namedtuple
from numba.core import types, ir
from numba.core.typing import signature
def assign_inplace(self, rhs, typ, name) -> ir.Var:
    """Assign a value to a new variable or inplace if it already exist

        Parameters
        ----------
        rhs : object
            The value
        typ : types.Type
            type of the value
        name : str
            variable name to store to

        Returns
        -------
        res : ir.Var
        """
    loc = self._loc
    var = ir.Var(self._scope, name, loc)
    assign = ir.Assign(rhs, var, loc)
    self._typemap.setdefault(var.name, typ)
    self._lowerer.lower_inst(assign)
    return var