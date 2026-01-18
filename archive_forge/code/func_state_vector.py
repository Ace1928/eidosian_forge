from pathlib import Path
from typing import List, Sequence
from warnings import warn
import numpy as np
from pennylane_lightning.core.lightning_base import (
@property
def state_vector(self):
    """Returns a handle to the statevector."""
    return self._qubit_state