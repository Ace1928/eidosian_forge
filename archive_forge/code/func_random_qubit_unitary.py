import itertools
from typing import Union, Sequence, Optional
import numpy as np
from cirq.value import random_state
def random_qubit_unitary(shape: Sequence[int]=(), randomize_global_phase: bool=False, rng: Optional[np.random.RandomState]=None) -> np.ndarray:
    """Random qubit unitary distributed over the Haar measure.

    The implementation is vectorized for speed.

    Args:
        shape: The broadcasted shape of the output. This is used to generate
            a tensor of random unitaries with dimensions tuple(shape) + (2,2).
        randomize_global_phase: (Default False) If True, a global phase is also
            sampled randomly. This corresponds to sampling over U(2) instead of
            SU(2).
        rng: Random number generator to be used in sampling. Default is
            numpy.random.
    """
    real_rng = random_state.parse_random_state(rng)
    theta = np.arcsin(np.sqrt(real_rng.rand(*shape)))
    phi_d = real_rng.rand(*shape) * np.pi * 2
    phi_o = real_rng.rand(*shape) * np.pi * 2
    out = _single_qubit_unitary(theta, phi_d, phi_o)
    if randomize_global_phase:
        out = np.moveaxis(out, (-2, -1), (0, 1))
        out *= np.exp(1j * np.pi * 2 * real_rng.rand(*shape))
        out = np.moveaxis(out, (0, 1), (-2, -1))
    return out