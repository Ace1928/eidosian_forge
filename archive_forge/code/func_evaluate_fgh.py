import logging
import types
import weakref
from pyomo.common.pyomo_typing import overload
from ctypes import (
from pyomo.common.autoslots import AutoSlots
from pyomo.common.fileutils import find_library
from pyomo.core.expr.numvalue import (
import pyomo.core.expr as EXPR
from pyomo.core.base.component import Component
from pyomo.core.base.units_container import units
def evaluate_fgh(self, args, fixed=None, fgh=2):
    """Evaluate the function and gradients given the specified arguments

        This evaluates the function given the specified arguments
        returning a 3-tuple of (function value [f], list of first partial
        derivatives [g], and the upper triangle of the Hessian matrix
        [h]).

        Parameters
        ----------
        args: Iterable
            Iterable containing the arguments to pass to the external
            function.  Non-native type elements will be converted to a
            native value using the :py:func:`value()` function.

        fixed: Optional[List[bool]]
            List of values indicating if the corresponding argument
            value is fixed.  Any fixed indices are guaranteed to return
            0 for first and second derivatives, regardless of what is
            computed by the external function.

        fgh: {0, 1, 2}
            What evaluations to return:

            * **0**: just return function evaluation
            * **1**: return function and first derivatives
            * **2**: return function, first derivatives, and hessian matrix

            Any return values not requested will be `None`.

        Returns
        -------
        f: float
            The return value of the function evaluated at `args`
        g: List[float] or None
            The list of first partial derivatives
        h: List[float] or None
            The upper-triangle of the Hessian matrix (second partial
            derivatives), stored column-wise.  Element :math:`H_{i,j}`
            (with :math:`0 <= i <= j < N` are mapped using
            :math:`h[i + j*(j + 1)/2] == H_{i,j}`.

        """
    args_ = [arg if arg.__class__ in native_types else value(arg) for arg in args]
    N = len(args_)
    f, g, h = self._evaluate(args_, fixed, fgh)
    if fgh == 2:
        n = N - 1
        if len(h) - 1 != n + n * (n + 1) // 2:
            raise RuntimeError(f"External function '{self.name}' returned an invalid Hessian matrix (expected {n + n * (n + 1) // 2 + 1}, received {len(h)})")
    else:
        h = None
    if fgh >= 1:
        if len(g) != N:
            raise RuntimeError(f"External function '{self.name}' returned an invalid derivative vector (expected {N}, received {len(g)})")
    else:
        g = None
    if fixed is not None:
        if fgh >= 1:
            for i, v in enumerate(fixed):
                if not v:
                    continue
                g[i] = 0
        if fgh >= 2:
            for i, v in enumerate(fixed):
                if not v:
                    continue
                for j in range(N):
                    if i <= j:
                        h[i + j * (j + 1) // 2] = 0
                    else:
                        h[j + i * (i + 1) // 2] = 0
    return (f, g, h)