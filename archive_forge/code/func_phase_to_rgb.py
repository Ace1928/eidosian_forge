from typing import List, Union
from functools import reduce
import colorsys
import numpy as np
from qiskit import user_config
from qiskit.quantum_info.states.statevector import Statevector
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.symplectic import PauliList, SparsePauliOp
from qiskit.quantum_info.states.densitymatrix import DensityMatrix
from qiskit.utils import optionals as _optionals
from qiskit.circuit.tools.pi_check import pi_check
from .array import _num_to_latex, array_to_latex
from .utils import matplotlib_close_if_inline
from .exceptions import VisualizationError
def phase_to_rgb(complex_number):
    """Map a phase of a complexnumber to a color in (r,g,b).

    complex_number is phase is first mapped to angle in the range
    [0, 2pi] and then to the HSL color wheel
    """
    angles = (np.angle(complex_number) + np.pi * 5 / 4) % (np.pi * 2)
    rgb = colorsys.hls_to_rgb(angles / (np.pi * 2), 0.5, 0.5)
    return rgb