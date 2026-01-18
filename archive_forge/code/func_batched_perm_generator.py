from __future__ import annotations
import warnings
import numpy as np
from itertools import combinations, permutations, product
from collections.abc import Sequence
import inspect
from scipy._lib._util import check_random_state, _rename_parameter
from scipy.special import ndtr, ndtri, comb, factorial
from scipy._lib._util import rng_integers
from dataclasses import dataclass
from ._common import ConfidenceInterval
from ._axis_nan_policy import _broadcast_concatenate, _broadcast_arrays
from ._warnings_errors import DegenerateDataWarning
def batched_perm_generator():
    for k in range(0, n_permutations, batch):
        batch_actual = min(batch, n_permutations - k)
        size = (batch_actual, n_samples, n_obs_sample)
        x = random_state.random(size=size)
        yield np.argsort(x, axis=-1)[:batch_actual]