import sys
from collections import namedtuple
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING, Union, cast
import numpy as np
from pyquil.external.rpcq import CompilerISA
from pyquil.gates import I, RX, MEASURE
from pyquil.noise_gates import _get_qvm_noise_supported_gates
from pyquil.quilatom import MemoryReference, format_parameter, ParameterDesignator
from pyquil.quilbase import Pragma, Gate, Declare
def damping_after_dephasing(T1: float, T2: float, gate_time: float) -> List[np.ndarray]:
    """
    Generate the Kraus map corresponding to the composition
    of a dephasing channel followed by an amplitude damping channel.

    :param T1: The amplitude damping time
    :param T2: The dephasing time
    :param gate_time: The gate duration.
    :return: A list of Kraus operators.
    """
    assert T1 >= 0
    assert T2 >= 0
    if T1 != INFINITY:
        damping = damping_kraus_map(p=1 - np.exp(-float(gate_time) / float(T1)))
    else:
        damping = [np.eye(2)]
    if T2 != INFINITY:
        gamma_phi = float(gate_time) / float(T2)
        if T1 != INFINITY:
            if T2 > 2 * T1:
                raise ValueError('T2 is upper bounded by 2 * T1')
            gamma_phi -= float(gate_time) / float(2 * T1)
        dephasing = dephasing_kraus_map(p=0.5 * (1 - np.exp(-gamma_phi)))
    else:
        dephasing = [np.eye(2)]
    return combine_kraus_maps(damping, dephasing)