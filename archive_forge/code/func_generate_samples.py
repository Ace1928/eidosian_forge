from pathlib import Path
from typing import List, Sequence
from warnings import warn
import numpy as np
from pennylane_lightning.core.lightning_base import (
def generate_samples(self):
    """Generate samples

            Returns:
                array[int]: array of samples in binary representation with shape
                    ``(dev.shots, dev.num_wires)``
            """
    measurements = MeasurementsC64(self.state_vector) if self.use_csingle else MeasurementsC128(self.state_vector)
    if self._mcmc:
        return measurements.generate_mcmc_samples(len(self.wires), self._kernel_name, self._num_burnin, self.shots).astype(int, copy=False)
    return measurements.generate_samples(len(self.wires), self.shots).astype(int, copy=False)