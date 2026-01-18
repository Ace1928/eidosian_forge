import abc
import collections
from typing import (
import numpy as np
from cirq import circuits, ops, protocols, study, value, work
from cirq.sim.simulation_state_base import SimulationStateBase
def sample_from_amplitudes(self, circuit: 'cirq.AbstractCircuit', param_resolver: 'cirq.ParamResolver', seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE', repetitions: int=1, qubit_order: 'cirq.QubitOrderOrList'=ops.QubitOrder.DEFAULT) -> Dict[int, int]:
    """Uses amplitude simulation to sample from the given circuit.

        This implements the algorithm outlined by Bravyi, Gosset, and Liu in
        https://arxiv.org/abs/2112.08499 to more efficiently calculate samples
        given an amplitude-based simulator.

        Simulators which also implement SimulatesSamples or SimulatesFullState
        should prefer `run()` or `simulate()`, respectively, as this method
        only accelerates sampling for amplitude-based simulators.

        Args:
            circuit: The circuit to simulate.
            param_resolver: Parameters to run with the program.
            seed: Random state to use as a seed. This must be provided
                manually - if the simulator has its own seed, it will not be
                used unless it is passed as this argument.
            repetitions: The number of repetitions to simulate.
            qubit_order: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.

        Returns:
            A dict of bitstrings sampled from the final state of `circuit` to
            the number of occurrences of that bitstring.

        Raises:
            ValueError: if 'circuit' has non-unitary elements, as differences
                in behavior between sampling steps break this algorithm.
        """
    prng = value.parse_random_state(seed)
    qubits = ops.QubitOrder.as_qubit_order(qubit_order).order_for(circuit.all_qubits())
    base_circuit = circuits.Circuit((ops.I(q) for q in qubits)) + circuit.unfreeze()
    qmap = {q: i for i, q in enumerate(qubits)}
    current_samples = {(0,) * len(qubits): repetitions}
    solved_circuit = protocols.resolve_parameters(base_circuit, param_resolver)
    if not protocols.has_unitary(solved_circuit):
        raise ValueError('sample_from_amplitudes does not support non-unitary behavior.')
    if protocols.is_measurement(solved_circuit):
        raise ValueError('sample_from_amplitudes does not support intermediate measurement.')
    for m_id, moment in enumerate(solved_circuit[1:]):
        circuit_prefix = solved_circuit[:m_id + 1]
        for t, op in enumerate(moment.operations):
            new_samples: Dict[Tuple[int, ...], int] = collections.defaultdict(int)
            qubit_indices = {qmap[q] for q in op.qubits}
            subcircuit = circuit_prefix + circuits.Moment(moment.operations[:t + 1])
            for current_sample, count in current_samples.items():
                sample_set = [current_sample]
                for idx in qubit_indices:
                    sample_set = [target[:idx] + (result,) + target[idx + 1:] for target in sample_set for result in [0, 1]]
                bitstrings = [int(''.join(map(str, sample)), base=2) for sample in sample_set]
                amps = self.compute_amplitudes(subcircuit, bitstrings, qubit_order=qubit_order)
                weights = np.abs(np.square(np.array(amps))).astype(np.float64)
                weights /= np.linalg.norm(weights, 1)
                subsample = prng.choice(len(sample_set), p=weights, size=count)
                for sample_index in subsample:
                    new_samples[sample_set[sample_index]] += 1
            current_samples = new_samples
    return {int(''.join(map(str, k)), base=2): v for k, v in current_samples.items()}