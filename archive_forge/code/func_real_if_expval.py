from numbers import Number
from typing import Tuple
import numpy as np
import pennylane as qml
from pennylane.operation import operation_derivative
from pennylane.tape import QuantumTape
from .apply_operation import apply_operation
from .simulate import get_final_state
from .initialize_state import create_initial_state
def real_if_expval(val):
    return np.real(val)