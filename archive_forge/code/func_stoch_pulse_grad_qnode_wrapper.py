from typing import Sequence, Callable
from functools import partial
import warnings
import numpy as np
import pennylane as qml
from pennylane.pulse import ParametrizedEvolution, HardwareHamiltonian
from pennylane import transform
from .parameter_shift import _make_zero_rep
from .general_shift_rules import eigvals_to_frequencies, generate_shift_rule
from .gradient_transform import (
@stoch_pulse_grad.custom_qnode_transform
def stoch_pulse_grad_qnode_wrapper(self, qnode, targs, tkwargs):
    """A custom QNode wrapper for the gradient transform :func:`~.stoch_pulse_grad`.
    It raises an error, so that applying ``stoch_pulse_grad`` to a ``QNode`` directly
    is not supported.
    """
    transform_name = 'stochastic pulse parameter-shift'
    raise_pulse_diff_on_qnode(transform_name)