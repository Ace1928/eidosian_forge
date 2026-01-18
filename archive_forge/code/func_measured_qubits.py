import abc
import enum
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING
from typing_extensions import Self
from cirq.value import digits, value_equality_attr
@property
def measured_qubits(self) -> Mapping['cirq.MeasurementKey', List[Tuple['cirq.Qid', ...]]]:
    """Gets the a mapping from measurement key to the qubits measured."""
    return self._measured_qubits