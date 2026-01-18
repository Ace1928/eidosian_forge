import functools
import itertools
from collections.abc import Iterable, Sequence
import numpy as np
from pennylane.pytrees import register_pytree
def select_random(self, n_samples, seed=None):
    """
        Returns a randomly sampled subset of Wires of length 'n_samples'.

        Args:
            n_samples (int): number of subsampled wires
            seed (int): optional random seed used for selecting the wires

        Returns:
            Wires: random subset of wires
        """
    if n_samples > len(self._labels):
        raise WireError(f'Cannot sample {n_samples} wires from {len(self._labels)} wires.')
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(self._labels), size=n_samples, replace=False)
    subset = tuple((self[i] for i in indices))
    return Wires(subset, _override=True)