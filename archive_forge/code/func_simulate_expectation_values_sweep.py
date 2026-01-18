import abc
import collections
from typing import (
import numpy as np
from cirq import circuits, ops, protocols, study, value, work
from cirq.sim.simulation_state_base import SimulationStateBase
def simulate_expectation_values_sweep(self, program: 'cirq.AbstractCircuit', observables: Union['cirq.PauliSumLike', List['cirq.PauliSumLike']], params: 'cirq.Sweepable', qubit_order: 'cirq.QubitOrderOrList'=ops.QubitOrder.DEFAULT, initial_state: Any=None, permit_terminal_measurements: bool=False) -> List[List[float]]:
    """Wraps computed expectation values in a list.

        Prefer overriding `simulate_expectation_values_sweep_iter`.
        """
    return list(self.simulate_expectation_values_sweep_iter(program, observables, params, qubit_order, initial_state, permit_terminal_measurements))