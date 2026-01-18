from typing import Any, Callable, Hashable, Iterable, Optional, Union, overload
import attr
import cachetools
import cirq
from cirq_ft.infra.decompose_protocol import _decompose_once_considering_known_decomposition
from typing_extensions import Literal, Protocol
from cirq_ft.deprecation import deprecated_cirq_ft_class, deprecated_cirq_ft_function
@deprecated_cirq_ft_function()
def t_complexity(stc: Any, fail_quietly: bool=False) -> Optional[TComplexity]:
    """Returns the TComplexity.

    Args:
        stc: an object to compute its TComplexity.
        fail_quietly: bool whether to return None on failure or raise an error.

    Returns:
        The TComplexity of the given object or None on failure (and fail_quietly=True).

    Raises:
        TypeError: if fail_quietly=False and the methods fails to compute TComplexity.
    """
    if isinstance(stc, (cirq.Gate, cirq.Operation)) and isinstance(stc, Hashable):
        ret = _t_complexity_for_gate_or_op(stc, fail_quietly)
    else:
        strategies = [_has_t_complexity, _is_clifford_or_t, _from_decomposition, _is_iterable]
        ret = _t_complexity_from_strategies(stc, fail_quietly, strategies)
    if ret is None and (not fail_quietly):
        raise TypeError(f"couldn't compute TComplexity of:\ntype: {type(stc)}\nvalue: {stc}")
    return ret