import itertools
import types
import warnings
from collections import defaultdict
from typing import (
import numpy as np
from rpcq.messages import NativeQuilMetadata, ParameterAref
from pyquil._parser.parser import run_parser
from pyquil._memory import Memory
from pyquil.gates import MEASURE, RESET, MOVE
from pyquil.noise import _check_kraus_ops, _create_kraus_pragmas, pauli_kraus_map
from pyquil.quilatom import (
from pyquil.quilbase import (
from pyquil.quiltcalibrations import (
def get_default_qubit_mapping(program: Program) -> Dict[Union[Qubit, QubitPlaceholder], Qubit]:
    """
    Takes a program which contains qubit placeholders and provides a mapping to the integers
    0 through N-1.

    The output of this function is suitable for input to :py:func:`address_qubits`.

    :param program: A program containing qubit placeholders
    :return: A dictionary mapping qubit placeholder to an addressed qubit from 0 through N-1.
    """
    fake_qubits, real_qubits, qubits = _what_type_of_qubit_does_it_use(program)
    if real_qubits:
        warnings.warn("This program contains integer qubits, so getting a mapping doesn't make sense.")
        return {q: cast(Qubit, q) for q in qubits}
    return {qp: Qubit(i) for i, qp in enumerate(qubits)}