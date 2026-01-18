import dataclasses
import itertools
from typing import (
import networkx as nx
import numpy as np
from cirq import circuits, devices, ops, protocols, value
from cirq._doc import document
def get_random_combinations_for_pairs(n_library_circuits: int, n_combinations: int, all_pairs: List[List[QidPairT]], random_state: 'cirq.RANDOM_STATE_OR_SEED_LIKE'=None) -> List[CircuitLibraryCombination]:
    """For an explicit nested list of pairs, prepare a set of combinations to efficiently sample
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
        all_pairs: A nested list of qubit pairs. The outer list should represent a "layer"
            where the inner pairs should all be able to be activated simultaneously.
        random_state: A random-state-like object to seed the random combination generation.

    Returns:
        A list of `CircuitLibraryCombination`, each corresponding to an interaction
        layer the outer list of `all_pairs`. Each object has a `combinations` matrix of circuit
        indices of shape `(n_combinations, len(pairs))` where `len(pairs)` may
        be different for each entry. This
        returned list can be provided to `sample_2q_xeb_circuits` to efficiently
        sample parallel XEB circuits.
    """

    def pair_gen():
        for pairs in all_pairs:
            yield (pairs, None)
    return _get_random_combinations(n_library_circuits=n_library_circuits, n_combinations=n_combinations, random_state=random_state, pair_gen=pair_gen())