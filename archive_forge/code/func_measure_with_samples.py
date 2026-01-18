from typing import List, Union, Tuple
import numpy as np
import pennylane as qml
from pennylane.ops import Sum, Hamiltonian, SProd, Prod
from pennylane.measurements import (
from pennylane.typing import TensorLike
from .apply_operation import apply_operation
from .measure import flatten_state
def measure_with_samples(mps: List[Union[SampleMeasurement, ClassicalShadowMP, ShadowExpvalMP]], state: np.ndarray, shots: Shots, is_state_batched: bool=False, rng=None, prng_key=None) -> List[TensorLike]:
    """
    Returns the samples of the measurement process performed on the given state.
    This function assumes that the user-defined wire labels in the measurement process
    have already been mapped to integer wires used in the device.

    Args:
        mp (List[Union[SampleMeasurement, ClassicalShadowMP, ShadowExpvalMP]]):
            The sample measurements to perform
        state (np.ndarray[complex]): The state vector to sample from
        shots (Shots): The number of samples to take
        is_state_batched (bool): whether the state is batched or not
        rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]): A
            seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
            If no value is provided, a default RNG will be used.
        prng_key (Optional[jax.random.PRNGKey]): An optional ``jax.random.PRNGKey``. This is
            the key to the JAX pseudo random number generator. Only for simulation using JAX.

    Returns:
        List[TensorLike[Any]]: Sample measurement results
    """
    groups, indices = _group_measurements(mps)
    all_res = []
    for group in groups:
        if isinstance(group[0], ExpectationMP) and isinstance(group[0].obs, Hamiltonian):
            measure_fn = _measure_hamiltonian_with_samples
        elif isinstance(group[0], ExpectationMP) and isinstance(group[0].obs, Sum):
            measure_fn = _measure_sum_with_samples
        elif isinstance(group[0], (ClassicalShadowMP, ShadowExpvalMP)):
            measure_fn = _measure_classical_shadow
        else:
            measure_fn = _measure_with_samples_diagonalizing_gates
        all_res.extend(measure_fn(group, state, shots, is_state_batched=is_state_batched, rng=rng, prng_key=prng_key))
    flat_indices = [_i for i in indices for _i in i]
    sorted_res = tuple((res for _, res in sorted(list(enumerate(all_res)), key=lambda r: flat_indices[r[0]])))
    if shots.has_partitioned_shots:
        sorted_res = tuple(zip(*sorted_res))
    return sorted_res