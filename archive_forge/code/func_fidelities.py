import datetime
from typing import Optional, Set, Tuple
import cirq
def fidelities(self) -> dict:
    """Returns the metrics (fidelities)."""
    return self._calibration_dict['fidelity']