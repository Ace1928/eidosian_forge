import abc
import enum
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING
from typing_extensions import Self
from cirq.value import digits, value_equality_attr
def record_measurement(self, key: 'cirq.MeasurementKey', measurement: Sequence[int], qubits: Sequence['cirq.Qid']):
    if len(measurement) != len(qubits):
        raise ValueError(f'{len(measurement)} measurements but {len(qubits)} qubits.')
    if key not in self._measurement_types:
        self._measurement_types[key] = MeasurementType.MEASUREMENT
        self._records[key] = []
        self._measured_qubits[key] = []
    if self._measurement_types[key] != MeasurementType.MEASUREMENT:
        raise ValueError(f'Channel Measurement already logged to key {key}')
    measured_qubits = self._measured_qubits[key]
    if measured_qubits:
        shape = tuple((q.dimension for q in qubits))
        key_shape = tuple((q.dimension for q in measured_qubits[-1]))
        if shape != key_shape:
            raise ValueError(f'Measurement shape {shape} does not match {key_shape} in {key}.')
    measured_qubits.append(tuple(qubits))
    self._records[key].append(tuple(measurement))