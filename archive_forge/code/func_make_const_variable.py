from collections import namedtuple
from numba.core import types, ir
from numba.core.typing import signature
def make_const_variable(self, cval, typ, name='pf_const') -> ir.Var:
    """Makes a constant variable

        Parameters
        ----------
        cval : object
            The constant value
        typ : types.Type
            type of the value
        name : str
            variable name to store to

        Returns
        -------
        res : ir.Var
        """
    return self.assign(rhs=ir.Const(cval, loc=self._loc), typ=typ, name=name)