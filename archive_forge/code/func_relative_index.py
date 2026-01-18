import abc
from typing import Any, cast, Tuple, TYPE_CHECKING, Union, Dict
from cirq._doc import document
from cirq.ops import common_gates, raw_types, identity
from cirq.type_workarounds import NotImplementedType
def relative_index(self, second: 'Pauli') -> int:
    """Relative index of self w.r.t. second in the (X, Y, Z) cycle."""
    return (self._index - second._index + 1) % 3 - 1