import abc
from typing import Generic, Dict, Any, List, Sequence, Union
from unittest import mock
import duet
import numpy as np
import pytest
import cirq
from cirq import study
from cirq.sim.simulation_state import TSimulationState
from cirq.sim.simulator import (
def test_iter_definitions():
    mock_trial_result = SimulationTrialResult(params={}, measurements={}, final_simulator_state=[])

    class FakeNonIterSimulatorImpl(SimulatesAmplitudes, SimulatesExpectationValues, SimulatesFinalState):
        """A class which defines the non-Iterator simulator API methods.

        After v0.12, simulators are expected to implement the *_iter methods.
        """

        def compute_amplitudes_sweep(self, program: 'cirq.AbstractCircuit', bitstrings: Sequence[int], params: study.Sweepable, qubit_order: cirq.QubitOrderOrList=cirq.QubitOrder.DEFAULT) -> Sequence[Sequence[complex]]:
            return [[1.0]]

        def simulate_expectation_values_sweep(self, program: 'cirq.AbstractCircuit', observables: Union['cirq.PauliSumLike', List['cirq.PauliSumLike']], params: 'study.Sweepable', qubit_order: cirq.QubitOrderOrList=cirq.QubitOrder.DEFAULT, initial_state: Any=None, permit_terminal_measurements: bool=False) -> List[List[float]]:
            return [[1.0]]

        def simulate_sweep(self, program: 'cirq.AbstractCircuit', params: study.Sweepable, qubit_order: cirq.QubitOrderOrList=cirq.QubitOrder.DEFAULT, initial_state: Any=None) -> List[SimulationTrialResult]:
            return [mock_trial_result]
    non_iter_sim = FakeNonIterSimulatorImpl()
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.X(q0))
    bitstrings = [0]
    params = {}
    assert non_iter_sim.compute_amplitudes_sweep(circuit, bitstrings, params) == [[1.0]]
    amp_iter = non_iter_sim.compute_amplitudes_sweep_iter(circuit, bitstrings, params)
    assert next(amp_iter) == [1.0]
    obs = cirq.X(q0)
    assert non_iter_sim.simulate_expectation_values_sweep(circuit, obs, params) == [[1.0]]
    ev_iter = non_iter_sim.simulate_expectation_values_sweep_iter(circuit, obs, params)
    assert next(ev_iter) == [1.0]
    assert non_iter_sim.simulate_sweep(circuit, params) == [mock_trial_result]
    state_iter = non_iter_sim.simulate_sweep_iter(circuit, params)
    assert next(state_iter) == mock_trial_result