import tempfile
from typing import Iterable, Dict, List
import numpy as np
import pytest
import cirq
import cirq.work as cw
from cirq.work import _MeasurementSpec, BitstringAccumulator, group_settings_greedy, InitObsSetting
from cirq.work.observable_measurement import (
@pytest.mark.parametrize(('with_circuit_sweep', 'checkpoint'), [(True, True), (False, False)])
def test_measure_grouped_settings(with_circuit_sweep, checkpoint, tmpdir):
    qubits = cirq.LineQubit.range(1)
    q, = qubits
    tests = [(cirq.KET_ZERO, cirq.Z, 1), (cirq.KET_ONE, cirq.Z, -1), (cirq.KET_PLUS, cirq.X, 1), (cirq.KET_MINUS, cirq.X, -1), (cirq.KET_IMAG, cirq.Y, 1), (cirq.KET_MINUS_IMAG, cirq.Y, -1)]
    if with_circuit_sweep:
        ss = cirq.Linspace('a', 0, 1, 12)
    else:
        ss = None
    if checkpoint:
        checkpoint_fn = f'{tmpdir}/obs.json'
    else:
        checkpoint_fn = None
    for init, obs, coef in tests:
        setting = cw.InitObsSetting(init_state=init(q), observable=obs(q))
        grouped_settings = {setting: [setting]}
        circuit = cirq.Circuit(cirq.I.on_each(*qubits))
        results = cw.measure_grouped_settings(circuit=circuit, grouped_settings=grouped_settings, sampler=cirq.Simulator(), stopping_criteria=cw.RepetitionsStoppingCriteria(1000, repetitions_per_chunk=500), circuit_sweep=ss, checkpoint=CheckpointFileOptions(checkpoint=checkpoint, checkpoint_fn=checkpoint_fn))
        if with_circuit_sweep:
            for result in results:
                assert result.means() == [coef]
        else:
            result, = results
            assert result.means() == [coef]