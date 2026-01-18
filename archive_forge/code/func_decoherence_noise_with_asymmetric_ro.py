import sys
from collections import namedtuple
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING, Union, cast
import numpy as np
from pyquil.external.rpcq import CompilerISA
from pyquil.gates import I, RX, MEASURE
from pyquil.noise_gates import _get_qvm_noise_supported_gates
from pyquil.quilatom import MemoryReference, format_parameter, ParameterDesignator
from pyquil.quilbase import Pragma, Gate, Declare
def decoherence_noise_with_asymmetric_ro(isa: CompilerISA, p00: float=0.975, p11: float=0.911) -> NoiseModel:
    """Similar to :py:func:`_decoherence_noise_model`, but with asymmetric readout.

    For simplicity, we use the default values for T1, T2, gate times, et al. and only allow
    the specification of readout fidelities.
    """
    gates = _get_qvm_noise_supported_gates(isa)
    noise_model = _decoherence_noise_model(gates)
    aprobs = np.array([[p00, 1 - p00], [1 - p11, p11]])
    aprobs = {q: aprobs for q in noise_model.assignment_probs.keys()}
    return NoiseModel(noise_model.gates, aprobs)