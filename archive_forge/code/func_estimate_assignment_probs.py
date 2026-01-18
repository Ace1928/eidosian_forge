import sys
from collections import namedtuple
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING, Union, cast
import numpy as np
from pyquil.external.rpcq import CompilerISA
from pyquil.gates import I, RX, MEASURE
from pyquil.noise_gates import _get_qvm_noise_supported_gates
from pyquil.quilatom import MemoryReference, format_parameter, ParameterDesignator
from pyquil.quilbase import Pragma, Gate, Declare
def estimate_assignment_probs(q: int, trials: int, qc: 'PyquilApiQuantumComputer', p0: Optional['Program']=None) -> np.ndarray:
    """
    Estimate the readout assignment probabilities for a given qubit ``q``.
    The returned matrix is of the form::

            [[p00 p01]
             [p10 p11]]

    :param q: The index of the qubit.
    :param trials: The number of samples for each state preparation.
    :param qc: The quantum computer to sample from.
    :param p0: A header program to prepend to the state preparation programs. Will not be compiled by quilc, so it must
           be native Quil.
    :return: The assignment probability matrix
    """
    from pyquil.quil import Program
    if p0 is None:
        p0 = Program()
    p_i = (p0 + Program(Declare('ro', 'BIT', 1), I(q), MEASURE(q, MemoryReference('ro', 0)))).wrap_in_numshots_loop(trials)
    results_i = np.sum(_run(qc, p_i))
    p_x = (p0 + Program(Declare('ro', 'BIT', 1), RX(np.pi, q), MEASURE(q, MemoryReference('ro', 0)))).wrap_in_numshots_loop(trials)
    results_x = np.sum(_run(qc, p_x))
    p00 = 1.0 - results_i / float(trials)
    p11 = results_x / float(trials)
    return np.array([[p00, 1 - p11], [1 - p00, p11]])