import abc
import dataclasses
import itertools
import os
import tempfile
import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING, Union
import numpy as np
import pandas as pd
import sympy
from cirq import circuits, study, ops, value, protocols
from cirq._doc import document
from cirq.work.observable_grouping import group_settings_greedy, GROUPER_T
from cirq.work.observable_measurement_data import (
from cirq.work.observable_settings import InitObsSetting, observables_to_settings, _MeasurementSpec
def measure_grouped_settings(circuit: 'cirq.AbstractCircuit', grouped_settings: Dict[InitObsSetting, List[InitObsSetting]], sampler: 'cirq.Sampler', stopping_criteria: StoppingCriteria, *, readout_symmetrization: bool=False, circuit_sweep: 'cirq.Sweepable'=None, readout_calibrations: Optional[BitstringAccumulator]=None, checkpoint: CheckpointFileOptions=CheckpointFileOptions()) -> List[BitstringAccumulator]:
    """Measure a suite of grouped InitObsSetting settings.

    This is a low-level API for accessing the observable measurement
    framework. See also `measure_observables` and `measure_observables_df`.

    Args:
        circuit: The circuit. This can contain parameters, in which case
            you should also specify `circuit_sweep`.
        grouped_settings: A series of setting groups expressed as a dictionary.
            The key is the max-weight setting used for preparing single-qubit
            basis-change rotations. The value is a list of settings
            compatible with the maximal setting you desire to measure.
            Automated routing algorithms like `group_settings_greedy` can
            be used to construct this input.
        sampler: A sampler.
        stopping_criteria: A StoppingCriteria object that can report
            whether enough samples have been sampled.
        readout_symmetrization: If set to True, each `meas_spec` will be
            split into two runs: one normal and one where a bit flip is
            incorporated prior to measurement. In the latter case, the
            measured bit will be flipped back classically and accumulated
            together. This causes readout error to appear symmetric,
            p(0|0) = p(1|1).
        circuit_sweep: Additional parameter sweeps for parameters contained
            in `circuit`. The total sweep is the product of the circuit sweep
            with parameter settings for the single-qubit basis-change rotations.
        readout_calibrations: The result of `calibrate_readout_error`.
        checkpoint: Options to set up optional checkpointing of intermediate
            data for each iteration of the sampling loop. See the documentation
            for `CheckpointFileOptions` for more. Load in these results with
            `cirq.read_json`.

    Raises:
        ValueError: If readout calibration is specified, but `readout_symmetrization
            is not True.
    """
    if readout_calibrations is not None and (not readout_symmetrization):
        raise ValueError('Readout calibration only works if `readout_symmetrization` is enabled.')
    qubits = sorted({q for ms in grouped_settings.keys() for q in ms.init_state.qubits})
    qubit_to_index = {q: i for i, q in enumerate(qubits)}
    needs_init_layer = _needs_init_layer(grouped_settings)
    measurement_param_circuit = _with_parameterized_layers(circuit, qubits, needs_init_layer)
    accumulators = {}
    meas_specs_todo = []
    for max_setting, param_resolver in itertools.product(grouped_settings.keys(), study.to_resolvers(circuit_sweep)):
        circuit_params = param_resolver.param_dict
        meas_spec = _MeasurementSpec(max_setting=max_setting, circuit_params=circuit_params)
        accumulator = BitstringAccumulator(meas_spec=meas_spec, simul_settings=grouped_settings[max_setting], qubit_to_index=qubit_to_index, readout_calibration=readout_calibrations)
        accumulators[meas_spec] = accumulator
        meas_specs_todo += [meas_spec]
    while True:
        meas_specs_todo, repetitions = _check_meas_specs_still_todo(meas_specs=meas_specs_todo, accumulators=accumulators, stopping_criteria=stopping_criteria)
        if len(meas_specs_todo) == 0:
            break
        flippy_meas_specs, repetitions = _subdivide_meas_specs(meas_specs=meas_specs_todo, repetitions=repetitions, qubits=qubits, readout_symmetrization=readout_symmetrization)
        resolved_params = [flippy_ms.param_tuples(needs_init_layer=needs_init_layer) for flippy_ms in flippy_meas_specs]
        resolved_params = _to_sweep(resolved_params)
        results = sampler.run_sweep(program=measurement_param_circuit, params=resolved_params, repetitions=repetitions)
        assert len(results) == len(flippy_meas_specs), 'Not as many results received as sweeps requested!'
        for flippy_ms, result in zip(flippy_meas_specs, results):
            accumulator = accumulators[flippy_ms.meas_spec]
            bitstrings = np.logical_xor(flippy_ms.flips, result.measurements['z'])
            accumulator.consume_results(bitstrings.astype(np.uint8, casting='safe'))
        checkpoint.maybe_to_json(list(accumulators.values()))
    return list(accumulators.values())