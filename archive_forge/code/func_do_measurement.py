from typing import Any, List, Optional, Sequence, Tuple, Union, cast
import numpy as np
from numpy.random.mtrand import RandomState
from pyquil.paulis import PauliTerm, PauliSum
from pyquil.pyqvm import AbstractQuantumSimulator
from pyquil.quilbase import Gate
from pyquil.simulation.matrices import QUANTUM_GATES
from pyquil.simulation.tools import all_bitstrings
def do_measurement(self, qubit: int) -> int:
    """
        Measure a qubit, collapse the wavefunction, and return the measurement result.

        :param qubit: Index of the qubit to measure.
        :return: measured bit
        """
    if self.rs is None:
        raise ValueError('You have tried to perform a stochastic operation without setting the random state of the simulator. Might I suggest using a PyQVM object?')
    measurement_probs = get_measure_probabilities(self.wf, qubit)
    measured_bit = int(np.argmax(self.rs.uniform() < np.cumsum(measurement_probs)))
    other_bit = (measured_bit + 1) % 2
    other_bit_indices = (slice(None),) * qubit + (other_bit,) + (slice(None),) * (self.n_qubits - qubit - 1)
    self.wf[other_bit_indices] = 0
    meas_bit_indices = (slice(None),) * qubit + (measured_bit,) + (slice(None),) * (self.n_qubits - qubit - 1)
    self.wf[meas_bit_indices] /= np.sqrt(measurement_probs[measured_bit])
    return measured_bit