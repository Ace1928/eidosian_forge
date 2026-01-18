from typing import (
from cirq import protocols, value
from cirq.ops import (
@property
def mkey(self) -> 'cirq.MeasurementKey':
    return self._mkey