import copy
import functools
import inspect
import warnings
from collections.abc import Sequence
from typing import Union
import logging
import pennylane as qml
from pennylane import Device
from pennylane.measurements import CountsMP, MidMeasureMP, Shots
from pennylane.tape import QuantumTape, QuantumScript
from .execution import INTERFACE_MAP, SUPPORTED_INTERFACES
from .set_shots import set_shots
@staticmethod
def get_gradient_fn(device, interface, diff_method='best', shots=None):
    """Determine the best differentiation method, interface, and device
        for a requested device, interface, and diff method.

        Args:
            device (.Device): PennyLane device
            interface (str): name of the requested interface
            diff_method (str or .TransformDispatcher): The requested method of differentiation.
                If a string, allowed options are ``"best"``, ``"backprop"``, ``"adjoint"``,
                ``"device"``, ``"parameter-shift"``, ``"hadamard"``, ``"finite-diff"``, or ``"spsa"``.
                A gradient transform may also be passed here.

        Returns:
            tuple[str or .TransformDispatcher, dict, .Device: Tuple containing the ``gradient_fn``,
            ``gradient_kwargs``, and the device to use when calling the execute function.
        """
    if diff_method == 'best':
        return QNode.get_best_method(device, interface, shots=shots)
    if diff_method == 'backprop':
        return QNode._validate_backprop_method(device, interface, shots=shots)
    if diff_method == 'adjoint':
        return QNode._validate_adjoint_method(device)
    if diff_method == 'device':
        return QNode._validate_device_method(device)
    if diff_method == 'parameter-shift':
        return QNode._validate_parameter_shift(device)
    if diff_method == 'finite-diff':
        return (qml.gradients.finite_diff, {}, device)
    if diff_method == 'spsa':
        return (qml.gradients.spsa_grad, {}, device)
    if diff_method == 'hadamard':
        return (qml.gradients.hadamard_grad, {}, device)
    if isinstance(diff_method, str):
        raise qml.QuantumFunctionError(f"Differentiation method {diff_method} not recognized. Allowed options are ('best', 'parameter-shift', 'backprop', 'finite-diff', 'device', 'adjoint', 'spsa', 'hadamard').")
    if isinstance(diff_method, qml.transforms.core.TransformDispatcher):
        return (diff_method, {}, device)
    raise qml.QuantumFunctionError(f'Differentiation method {diff_method} must be a gradient transform or a string.')