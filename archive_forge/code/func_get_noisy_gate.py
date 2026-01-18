import sys
from collections import namedtuple
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING, Union, cast
import numpy as np
from pyquil.external.rpcq import CompilerISA
from pyquil.gates import I, RX, MEASURE
from pyquil.noise_gates import _get_qvm_noise_supported_gates
from pyquil.quilatom import MemoryReference, format_parameter, ParameterDesignator
from pyquil.quilbase import Pragma, Gate, Declare
def get_noisy_gate(gate_name: str, params: Iterable[ParameterDesignator]) -> Tuple[np.ndarray, str]:
    """
    Look up the numerical gate representation and a proposed 'noisy' name.

    :param gate_name: The Quil gate name
    :param params: The gate parameters.
    :return: A tuple (matrix, noisy_name) with the representation of the ideal gate matrix
        and a proposed name for the noisy version.
    """
    params = tuple(params)
    if gate_name == 'I':
        assert params == ()
        return (np.eye(2), 'NOISY-I')
    if gate_name == 'RX':
        angle, = params
        if not isinstance(angle, (int, float, complex)):
            raise TypeError(f'Cannot produce noisy gate for parameter of type {type(angle)}')
        if np.isclose(angle, np.pi / 2, atol=ANGLE_TOLERANCE):
            return (np.array([[1, -1j], [-1j, 1]]) / np.sqrt(2), 'NOISY-RX-PLUS-90')
        elif np.isclose(angle, -np.pi / 2, atol=ANGLE_TOLERANCE):
            return (np.array([[1, 1j], [1j, 1]]) / np.sqrt(2), 'NOISY-RX-MINUS-90')
        elif np.isclose(angle, np.pi, atol=ANGLE_TOLERANCE):
            return (np.array([[0, -1j], [-1j, 0]]), 'NOISY-RX-PLUS-180')
        elif np.isclose(angle, -np.pi, atol=ANGLE_TOLERANCE):
            return (np.array([[0, 1j], [1j, 0]]), 'NOISY-RX-MINUS-180')
    elif gate_name == 'CZ':
        assert params == ()
        return (np.diag([1, 1, 1, -1]), 'NOISY-CZ')
    raise NoisyGateUndefined('Undefined gate and params: {}{}\nPlease restrict yourself to I, RX(+/-pi), RX(+/-pi/2), CZ'.format(gate_name, params))