import warnings
from dataclasses import dataclass
from typing import Callable, List, Union
import numpy as np
import pennylane as qml
from pennylane.pulse import HardwareHamiltonian
from pennylane.pulse.hardware_hamiltonian import HardwarePulse
from pennylane.typing import TensorLike
from pennylane.wires import Wires
def callable_freq_to_angular(fn):
    """Add a factor of 2pi to a callable result to convert from Hz to rad/s"""

    def angular_fn(p, t):
        return 2 * np.pi * fn(p, t)
    return angular_fn