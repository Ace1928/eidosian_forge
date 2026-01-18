import dataclasses
from typing import Any, Dict, List, Sequence, Set, Type, TypeVar, Union
import numpy as np
import cirq, cirq_google
from cirq import _compat, devices
from cirq.devices import noise_utils
from cirq.transformers.heuristic_decompositions import gate_tabulation_math_utils
def extract_entangling_error(match_id: noise_utils.OpIdentifier):
    """Gets the entangling error component of depol_errors[match_id]."""
    unitary_err = cirq.unitary(self.fsim_errors[match_id])
    fid = gate_tabulation_math_utils.unitary_entanglement_fidelity(unitary_err, np.eye(4))
    return 1 - fid