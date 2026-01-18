import sys
from collections import namedtuple
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING, Union, cast
import numpy as np
from pyquil.external.rpcq import CompilerISA
from pyquil.gates import I, RX, MEASURE
from pyquil.noise_gates import _get_qvm_noise_supported_gates
from pyquil.quilatom import MemoryReference, format_parameter, ParameterDesignator
from pyquil.quilbase import Pragma, Gate, Declare
def append_kraus_to_gate(kraus_ops: Sequence[np.ndarray], gate_matrix: np.ndarray) -> List[Union[np.number, np.ndarray]]:
    """
    Follow a gate ``gate_matrix`` by a Kraus map described by ``kraus_ops``.

    :param kraus_ops: The Kraus operators.
    :param gate_matrix: The unitary gate.
    :return: A list of transformed Kraus operators.
    """
    return [kj.dot(gate_matrix) for kj in kraus_ops]