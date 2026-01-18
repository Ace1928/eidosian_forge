from typing import (
import numpy as np
import cirq
from cirq import value
from cirq_google.calibration.phased_fsim import (
def sample_gate(a: cirq.Qid, b: cirq.Qid, gate: cirq.FSimGate) -> PhasedFSimCharacterization:
    pair_parameters = None
    swapped = False
    if (a, b) in parameters:
        pair_parameters = parameters[a, b].get(gate)
    elif (b, a) in parameters:
        pair_parameters = parameters[b, a].get(gate)
        swapped = True
    if pair_parameters is None:
        raise ValueError(f'Missing parameters for value for pair {(a, b)} and gate {gate}.')
    if not isinstance(pair_parameters, PhasedFSimCharacterization):
        pair_parameters = PhasedFSimCharacterization(**pair_parameters)
    if swapped:
        pair_parameters = pair_parameters.parameters_for_qubits_swapped()
    return pair_parameters