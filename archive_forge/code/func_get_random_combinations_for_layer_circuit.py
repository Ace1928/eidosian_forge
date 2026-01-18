import dataclasses
import itertools
from typing import (
import networkx as nx
import numpy as np
from cirq import circuits, devices, ops, protocols, value
from cirq._doc import document
def get_random_combinations_for_layer_circuit(n_library_circuits: int, n_combinations: int, layer_circuit: 'cirq.Circuit', random_state: 'cirq.RANDOM_STATE_OR_SEED_LIKE'=None) -> List[CircuitLibraryCombination]:
    """For a layer circuit, prepare a set of combinations to efficiently sample
    parallel two-qubit XEB circuits.

    Args:
        n_library_circuits: The number of circuits in your library. Likely the value
            passed to `generate_library_of_2q_circuits`.
        n_combinations: The number of combinations (with replacement) to generate
            using the library circuits. Since this function returns a
            `CircuitLibraryCombination`, the combinations will be represented
            by indexes between 0 and `n_library_circuits-1` instead of the circuits
            themselves. The more combinations, the more precise of an estimate for XEB
            fidelity estimation, but a corresponding increase in the number of circuits
            you must sample.
        layer_circuit: A calibration-style circuit where each Moment represents a layer.
            Two qubit operations indicate the pair should be activated. This circuit should
            only contain Moments which only contain two-qubit operations.
        random_state: A random-state-like object to seed the random combination generation.

    Returns:
        A list of `CircuitLibraryCombination`, each corresponding to a moment in `layer_circuit`.
        Each object has a `combinations` matrix of circuit
        indices of shape `(n_combinations, len(pairs))` where `len(pairs)` may
        be different for each entry (i.e. for moment). This
        returned list can be provided to `sample_2q_xeb_circuits` to efficiently
        sample parallel XEB circuits.
    """

    def pair_gen():
        for moment in layer_circuit.moments:
            yield (_pairs_from_moment(moment), moment)
    return _get_random_combinations(n_library_circuits=n_library_circuits, n_combinations=n_combinations, random_state=random_state, pair_gen=pair_gen())