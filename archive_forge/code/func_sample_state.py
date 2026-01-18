from typing import List, Union, Tuple
import numpy as np
import pennylane as qml
from pennylane.ops import Sum, Hamiltonian, SProd, Prod
from pennylane.measurements import (
from pennylane.typing import TensorLike
from .apply_operation import apply_operation
from .measure import flatten_state
def sample_state(state, shots: int, is_state_batched: bool=False, wires=None, rng=None, prng_key=None) -> np.ndarray:
    """
    Returns a series of samples of a state.

    Args:
        state (array[complex]): A state vector to be sampled
        shots (int): The number of samples to take
        is_state_batched (bool): whether the state is batched or not
        wires (Sequence[int]): The wires to sample
        rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]):
            A seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
            If no value is provided, a default RNG will be used
        prng_key (Optional[jax.random.PRNGKey]): An optional ``jax.random.PRNGKey``. This is
            the key to the JAX pseudo random number generator. Only for simulation using JAX.

    Returns:
        ndarray[int]: Sample values of the shape (shots, num_wires)
    """
    if prng_key is not None:
        return _sample_state_jax(state, shots, prng_key, is_state_batched=is_state_batched, wires=wires)
    rng = np.random.default_rng(rng)
    total_indices = len(state.shape) - is_state_batched
    state_wires = qml.wires.Wires(range(total_indices))
    wires_to_sample = wires or state_wires
    num_wires = len(wires_to_sample)
    basis_states = np.arange(2 ** num_wires)
    flat_state = flatten_state(state, total_indices)
    with qml.queuing.QueuingManager.stop_recording():
        probs = qml.probs(wires=wires_to_sample).process_state(flat_state, state_wires)
    if is_state_batched:
        samples = np.stack([rng.choice(basis_states, shots, p=p) for p in probs])
    else:
        samples = rng.choice(basis_states, shots, p=probs)
    powers_of_two = 1 << np.arange(num_wires, dtype=np.int64)[::-1]
    states_sampled_base_ten = samples[..., None] & powers_of_two
    return (states_sampled_base_ten > 0).astype(np.int64)