import abc
import enum
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING
from typing_extensions import Self
from cirq.value import digits, value_equality_attr
def record_channel_measurement(self, key: 'cirq.MeasurementKey', measurement: int):
    if key not in self._measurement_types:
        self._measurement_types[key] = MeasurementType.CHANNEL
        self._channel_records[key] = []
    if self._measurement_types[key] != MeasurementType.CHANNEL:
        raise ValueError(f'Measurement already logged to key {key}')
    self._channel_records[key].append(measurement)