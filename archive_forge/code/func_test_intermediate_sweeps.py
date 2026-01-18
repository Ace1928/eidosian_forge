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
@mock.patch.multiple(SimulatesIntermediateStateImpl, __abstractmethods__=set(), simulate_moment_steps=mock.Mock())
def test_intermediate_sweeps():
    simulator = SimulatesIntermediateStateImpl()
    final_state = np.array([1, 0, 0, 0])

    def steps(*args, **kwargs):
        result = mock.Mock()
        result.measurements = {'a': np.array([True, True])}
        result._simulator_state.return_value = final_state
        yield result
    simulator.simulate_moment_steps.side_effect = steps
    circuit = mock.Mock(cirq.Circuit)
    param_resolvers = [cirq.ParamResolver({}), cirq.ParamResolver({})]
    qubit_order = mock.Mock(cirq.QubitOrder)
    results = simulator.simulate_sweep(program=circuit, params=param_resolvers, qubit_order=qubit_order, initial_state=2)
    expected_results = [cirq.SimulationTrialResult(measurements={'a': np.array([True, True])}, params=param_resolvers[0], final_simulator_state=final_state), cirq.SimulationTrialResult(measurements={'a': np.array([True, True])}, params=param_resolvers[1], final_simulator_state=final_state)]
    assert results == expected_results