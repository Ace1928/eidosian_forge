from typing import Any, List, Optional, Sequence, Tuple, Union, cast
import numpy as np
from numpy.random.mtrand import RandomState
from pyquil.paulis import PauliTerm, PauliSum
from pyquil.pyqvm import AbstractQuantumSimulator
from pyquil.quilbase import Gate
from pyquil.simulation.matrices import QUANTUM_GATES
from pyquil.simulation.tools import all_bitstrings
def do_post_gate_noise(self, noise_type: str, noise_prob: float, qubits: List[int]) -> 'AbstractQuantumSimulator':
    raise NotImplementedError('The numpy simulator cannot handle noise')