import dataclasses
from typing import Union, Iterable, TYPE_CHECKING
from cirq import circuits, study, ops
from cirq.work.observable_measurement import measure_grouped_settings, StoppingCriteria
from cirq.work.observable_settings import InitObsSetting, zeros_state
def calibrate_readout_error(qubits: Iterable[ops.Qid], sampler: Union['cirq.Simulator', 'cirq.Sampler'], stopping_criteria: StoppingCriteria):
    stopping_criteria = dataclasses.replace(stopping_criteria, repetitions_per_chunk=100000)
    init_state = zeros_state(qubits)
    max_setting = InitObsSetting(init_state=init_state, observable=ops.PauliString({q: ops.Z for q in qubits}))
    grouped_settings = {max_setting: [InitObsSetting(init_state=init_state, observable=ops.PauliString({q: ops.Z})) for q in qubits]}
    results = measure_grouped_settings(circuit=circuits.Circuit(), grouped_settings=grouped_settings, sampler=sampler, stopping_criteria=stopping_criteria, circuit_sweep=study.UnitSweep, readout_symmetrization=True)
    result, = list(results)
    return result