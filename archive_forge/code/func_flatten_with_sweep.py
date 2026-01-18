from typing import Any, Callable, List, Optional, Tuple, Union, TYPE_CHECKING
import numbers
import sympy
from cirq import protocols
from cirq.study import resolver, sweeps, sweepable
def flatten_with_sweep(val: Any, sweep: Union[sweeps.Sweep, List[resolver.ParamResolver]]) -> Tuple[Any, sweeps.Sweep]:
    """Creates a copy of `val` with any symbols or expressions replaced with
    new symbols.  `val` can be a `Circuit`, `Gate`, `Operation`, or other
    type.  Also transforms a sweep over the symbols in `val` to a sweep over the
    new symbols.

    `flatten_with_sweep` goes through every parameter in `val` and does the
    following:
    - If the parameter is a number, don't change it.
    - If the parameter is a symbol, don't change it and use the same symbol with
        the same values in the new sweep.
    - If the parameter is an expression, replace it with a symbol and use the
        new symbol with the evaluated value of the expression in the new sweep.
        The new symbol will be `sympy.Symbol('<x + 1>')` if the expression was
        `sympy.Symbol('x') + 1`.  In the unlikely case that an expression with a
        different meaning also has the string `'x + 1'`, a number is appended to
        the name to avoid collision: `sympy.Symbol('<x + 1>_1')`.

    Args:
        val: The value to copy and substitute parameter expressions with
        flattened symbols.
        sweep: A sweep over parameters used by `val`.

    Returns:
        The tuple (new value, new sweep) where new value is `val` with flattened
        expressions and new sweep is the equivalent sweep over it.
    """
    val_flat, expr_map = flatten(val)
    new_sweep = expr_map.transform_sweep(sweep)
    return (val_flat, new_sweep)