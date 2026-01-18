import abc
from typing import Any, cast, Tuple, TYPE_CHECKING, Union, Dict
from cirq._doc import document
from cirq.ops import common_gates, raw_types, identity
from cirq.type_workarounds import NotImplementedType
@staticmethod
def by_index(index: int) -> 'Pauli':
    return Pauli._XYZ[index % 3]