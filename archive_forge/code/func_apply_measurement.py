from typing import Any, Dict, List, Sequence, Union
import numpy as np
import cirq
from cirq import protocols, value
from cirq.protocols import act_on
from cirq.sim import clifford, simulator_base
def apply_measurement(self, op: 'cirq.Operation', measurements: Dict[str, List[int]], prng: np.random.RandomState, collapse_state_vector=True):
    if not isinstance(op.gate, cirq.MeasurementGate):
        raise TypeError(f'apply_measurement only supports cirq.MeasurementGate operations. Found {op.gate} instead.')
    if collapse_state_vector:
        state = self
    else:
        state = self.copy()
    classical_data = value.ClassicalDataDictionaryStore()
    ch_form_args = clifford.StabilizerChFormSimulationState(prng=prng, classical_data=classical_data, qubits=self.qubit_map.keys(), initial_state=state.ch_form)
    act_on(op, ch_form_args)
    measurements.update({str(k): list(v[-1]) for k, v in classical_data.records.items()})