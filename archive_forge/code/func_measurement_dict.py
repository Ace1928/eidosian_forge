import collections
from typing import Dict, Counter, List, Optional, Sequence
import numpy as np
import cirq
def measurement_dict(self) -> Dict[str, Sequence[int]]:
    """Returns a map from measurement keys to target qubit indices for this measurement."""
    return self._measurement_dict