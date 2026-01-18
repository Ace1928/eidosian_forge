import dataclasses
import itertools
from typing import (
import networkx as nx
import numpy as np
from cirq import circuits, devices, ops, protocols, value
from cirq._doc import document
def random_rotations_between_grid_interaction_layers_circuit(qubits: Iterable['cirq.GridQubit'], depth: int, *, two_qubit_op_factory: Callable[['cirq.GridQubit', 'cirq.GridQubit', 'np.random.RandomState'], 'cirq.OP_TREE']=lambda a, b, _: ops.CZPowGate()(a, b), pattern: Sequence[GridInteractionLayer]=GRID_STAGGERED_PATTERN, single_qubit_gates: Sequence['cirq.Gate']=(ops.X ** 0.5, ops.Y ** 0.5, ops.PhasedXPowGate(phase_exponent=0.25, exponent=0.5)), add_final_single_qubit_layer: bool=True, seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE'=None) -> 'cirq.Circuit':
    """Generate a random quantum circuit of a particular form.

    This construction is based on the circuits used in the paper
    https://www.nature.com/articles/s41586-019-1666-5.

    The generated circuit consists of a number of "cycles", this number being
    specified by `depth`. Each cycle is actually composed of two sub-layers:
    a layer of single-qubit gates followed by a layer of two-qubit gates,
    controlled by their respective arguments, see below. The pairs of qubits
    in a given entangling layer is controlled by the `pattern` argument,
    see below.

    Args:
        qubits: The qubits to use.
        depth: The number of cycles.
        two_qubit_op_factory: A callable that returns a two-qubit operation.
            These operations will be generated with calls of the form
            `two_qubit_op_factory(q0, q1, prng)`, where `prng` is the
            pseudorandom number generator.
        pattern: A sequence of GridInteractionLayers, each of which determine
            which pairs of qubits are entangled. The layers in a pattern are
            iterated through sequentially, repeating until `depth` is reached.
        single_qubit_gates: Single-qubit gates are selected randomly from this
            sequence. No qubit is acted upon by the same single-qubit gate in
            consecutive cycles. If only one choice of single-qubit gate is
            given, then this constraint is not enforced.
        add_final_single_qubit_layer: Whether to include a final layer of
            single-qubit gates after the last cycle.
        seed: A seed or random state to use for the pseudorandom number
            generator.
    """
    prng = value.parse_random_state(seed)
    qubits = list(qubits)
    coupled_qubit_pairs = _coupled_qubit_pairs(qubits)
    circuit = circuits.Circuit()
    previous_single_qubit_layer = circuits.Moment()
    single_qubit_layer_factory = _single_qubit_gates_arg_to_factory(single_qubit_gates=single_qubit_gates, qubits=qubits, prng=prng)
    for i in range(depth):
        single_qubit_layer = single_qubit_layer_factory.new_layer(previous_single_qubit_layer)
        circuit += single_qubit_layer
        two_qubit_layer = _two_qubit_layer(coupled_qubit_pairs, two_qubit_op_factory, pattern[i % len(pattern)], prng)
        circuit += two_qubit_layer
        previous_single_qubit_layer = single_qubit_layer
    if add_final_single_qubit_layer:
        circuit += single_qubit_layer_factory.new_layer(previous_single_qubit_layer)
    return circuit