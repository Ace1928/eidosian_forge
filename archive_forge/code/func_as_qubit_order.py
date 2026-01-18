from typing import Any, Callable, Iterable, Optional, Tuple, TypeVar, TYPE_CHECKING
from cirq.ops import raw_types
@staticmethod
def as_qubit_order(val: 'qubit_order_or_list.QubitOrderOrList') -> 'QubitOrder':
    """Converts a value into a basis.

        Args:
            val: An iterable or a basis.

        Returns:
            The basis implied by the value.

        Raises:
            ValueError: If `val` is not an iterable or a `QubitOrder`.
        """
    if isinstance(val, Iterable):
        return QubitOrder.explicit(val)
    if isinstance(val, QubitOrder):
        return val
    raise ValueError(f"Don't know how to interpret <{val}> as a Basis.")