from typing import Any, Iterable, Mapping
import networkx as nx
import cirq
from cirq_aqt import aqt_target_gateset
@property
def measurement_duration(self) -> 'cirq.DURATION_LIKE':
    """Return the maximum duration of the measurement operation."""
    return self._measurement_duration