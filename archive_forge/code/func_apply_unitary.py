from typing import Any, Dict, List, Sequence, Union
import numpy as np
import cirq
from cirq import protocols, value
from cirq.protocols import act_on
from cirq.sim import clifford, simulator_base
def apply_unitary(self, op: 'cirq.Operation'):
    ch_form_args = clifford.StabilizerChFormSimulationState(prng=np.random.RandomState(), qubits=self.qubit_map.keys(), initial_state=self.ch_form)
    try:
        act_on(op, ch_form_args)
    except TypeError:
        raise ValueError(f'{op.gate} cannot be run with Clifford simulator.')
    return