import tempfile
from typing import Iterable, Dict, List
import numpy as np
import pytest
import cirq
import cirq.work as cw
from cirq.work import _MeasurementSpec, BitstringAccumulator, group_settings_greedy, InitObsSetting
from cirq.work.observable_measurement import (
def test_measure_grouped_settings_read_checkpoint(tmpdir):
    qubits = cirq.LineQubit.range(1)
    q, = qubits
    setting = cw.InitObsSetting(init_state=cirq.KET_ZERO(q), observable=cirq.Z(q))
    grouped_settings = {setting: [setting]}
    circuit = cirq.Circuit(cirq.I.on_each(*qubits))
    with pytest.raises(ValueError, match='same filename.*'):
        _ = cw.measure_grouped_settings(circuit=circuit, grouped_settings=grouped_settings, sampler=cirq.Simulator(), stopping_criteria=cw.RepetitionsStoppingCriteria(1000, repetitions_per_chunk=500), checkpoint=CheckpointFileOptions(checkpoint=True, checkpoint_fn=f'{tmpdir}/obs.json', checkpoint_other_fn=f'{tmpdir}/obs.json'))
    _ = cw.measure_grouped_settings(circuit=circuit, grouped_settings=grouped_settings, sampler=cirq.Simulator(), stopping_criteria=cw.RepetitionsStoppingCriteria(1000, repetitions_per_chunk=500), checkpoint=CheckpointFileOptions(checkpoint=True, checkpoint_fn=f'{tmpdir}/obs.json', checkpoint_other_fn=f'{tmpdir}/obs.prev.json'))
    results = cirq.read_json(f'{tmpdir}/obs.json')
    result, = results
    assert result.n_repetitions == 1000
    assert result.means() == [1.0]