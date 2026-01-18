from typing import Callable, Sequence
from functools import partial
import numpy as np
import pennylane as qml
from pennylane.pulse import ParametrizedEvolution
from pennylane.ops.qubit.special_unitary import pauli_basis_strings, _pauli_decompose
from pennylane import transform
from .parameter_shift import _make_zero_rep
from .pulse_gradient import _assert_has_jax, raise_pulse_diff_on_qnode
from .gradient_transform import (
@pulse_odegen.custom_qnode_transform
def pulse_odegen_qnode_wrapper(self, qnode, targs, tkwargs):
    """A custom QNode wrapper for the gradient transform :func:`~.pulse_odegen`.
    It raises an error, so that applying ``pulse_odegen`` to a ``QNode`` directly
    is not supported.
    """
    transform_name = 'pulse generator parameter-shift'
    raise_pulse_diff_on_qnode(transform_name)