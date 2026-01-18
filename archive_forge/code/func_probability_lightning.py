from pathlib import Path
from typing import List, Sequence
from warnings import warn
import numpy as np
from pennylane_lightning.core.lightning_base import (
def probability_lightning(self, wires):
    """Return the probability of each computational basis state.

            Args:
                wires (Iterable[Number, str], Number, str, Wires): wires to return
                    marginal probabilities for. Wires not provided are traced out of the system.

            Returns:
                array[float]: list of the probabilities
            """
    return (MeasurementsC64(self.state_vector) if self.use_csingle else MeasurementsC128(self.state_vector)).probs(wires)