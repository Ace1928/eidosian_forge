from typing import Any, List, Optional, Sequence, Tuple, Union, cast
import numpy as np
from numpy.random.mtrand import RandomState
from pyquil.paulis import PauliTerm, PauliSum
from pyquil.pyqvm import AbstractQuantumSimulator
from pyquil.quilbase import Gate
from pyquil.simulation.matrices import QUANTUM_GATES
from pyquil.simulation.tools import all_bitstrings
def sample_bitstrings(self, n_samples: int) -> np.ndarray:
    """
        Sample bitstrings from the distribution defined by the wavefunction.

        Qubit 0 is at ``out[:, 0]``.

        :param n_samples: The number of bitstrings to sample
        :return: An array of shape (n_samples, n_qubits)
        """
    if self.rs is None:
        raise ValueError('You have tried to perform a stochastic operation without setting the random state of the simulator. Might I suggest using a PyQVM object?')
    probabilities = np.abs(self.wf.reshape(-1)) ** 2
    possible_bitstrings = all_bitstrings(self.n_qubits)
    inds = self.rs.choice(2 ** self.n_qubits, n_samples, p=probabilities)
    return possible_bitstrings[inds, :]