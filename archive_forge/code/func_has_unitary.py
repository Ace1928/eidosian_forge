from typing import Any, TypeVar, Optional
import numpy as np
from typing_extensions import Protocol
from cirq import qis
from cirq._doc import doc_private
from cirq.protocols import qid_shape_protocol
from cirq.protocols.apply_unitary_protocol import ApplyUnitaryArgs
from cirq.protocols.decompose_protocol import _try_decompose_into_operations_and_qubits
def has_unitary(val: Any, *, allow_decompose: bool=True) -> bool:
    """Determines whether the value has a unitary effect.

    Determines whether `val` has a unitary effect by attempting the following
    strategies:

    1. Try to use `val.has_unitary()`.
        Case a) Method not present or returns `NotImplemented`.
            Inconclusive.
        Case b) Method returns `True`.
            Unitary.
        Case c) Method returns `False`.
            Not unitary.

    2. Try to use `val._decompose_()`.
        Case a) Method not present or returns `NotImplemented` or `None`.
            Inconclusive.
        Case b) Method returns an OP_TREE containing only unitary operations.
            Unitary.
        Case c) Method returns an OP_TREE containing non-unitary operations.
            Not Unitary.

    3. Try to use `val._apply_unitary_(args)`.
        Case a) Method not present or returns `NotImplemented`.
            Inconclusive.
        Case b) Method returns a numpy array.
            Unitary.
        Case c) Method returns `None`.
            Not unitary.

    4. Try to use `val._unitary_()`.
        Case a) Method not present or returns `NotImplemented`.
            Continue to next strategy.
        Case b) Method returns a numpy array.
            Unitary.
        Case c) Method returns `None`.
            Not unitary.

    It is assumed that, when multiple of these strategies give a conclusive
    result, that these results will all be consistent with each other. If all
    strategies are inconclusive, the value is classified as non-unitary.

    Args:
        The value that may or may not have a unitary effect.

    Returns:
        Whether or not `val` has a unitary effect.
    """
    strats = [_strat_has_unitary_from_has_unitary, _strat_has_unitary_from_decompose, _strat_has_unitary_from_apply_unitary, _strat_has_unitary_from_unitary]
    if not allow_decompose:
        strats.remove(_strat_has_unitary_from_decompose)
    for strat in strats:
        result = strat(val)
        if result is not None:
            return result
    return False