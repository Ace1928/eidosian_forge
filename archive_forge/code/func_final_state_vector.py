from typing import (
import numpy as np
import cirq
from cirq import value
from cirq_google.calibration.phased_fsim import (
def final_state_vector(self, program: cirq.Circuit) -> np.ndarray:
    result = self.simulate(program)
    return result.state_vector()