from typing import Any, Callable, List, Optional, Tuple, Union, TYPE_CHECKING
import numbers
import sympy
from cirq import protocols
from cirq.study import resolver, sweeps, sweepable
@staticmethod
def default_get_param_name(val: sympy.Expr) -> str:
    if isinstance(val, sympy.Symbol):
        return val.name
    return f'<{val!s}>'