import abc
import collections
from typing import (
import numpy as np
from cirq import circuits, ops, protocols, study, value, work
from cirq.sim.simulation_state_base import SimulationStateBase
@value.alternative(requires='compute_amplitudes_sweep', implementation=_compute_amplitudes_sweep_to_iter)
def compute_amplitudes_sweep_iter(self, program: 'cirq.AbstractCircuit', bitstrings: Sequence[int], params: 'cirq.Sweepable', qubit_order: 'cirq.QubitOrderOrList'=ops.QubitOrder.DEFAULT) -> Iterator[Sequence[complex]]:
    """Computes the desired amplitudes.

        The initial state is assumed to be the all zeros state.

        Args:
            program: The circuit to simulate.
            bitstrings: The bitstrings whose amplitudes are desired, input
                as an integer array where each integer is formed from measured
                qubit values according to `qubit_order` from most to least
                significant qubit, i.e. in big-endian ordering. If inputting
                a binary literal add the prefix 0b or 0B.
                For example: 0010 can be input as 0b0010, 0B0010, 2, 0x2, etc.
            params: Parameters to run with the program.
            qubit_order: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.

        Returns:
            An Iterator over lists of amplitudes. The outer dimension indexes
            the circuit parameters and the inner dimension indexes bitstrings.
        """
    raise NotImplementedError()