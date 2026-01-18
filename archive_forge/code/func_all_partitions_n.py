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
def all_partitions_n(z, ns):
    if len(ns) == 0:
        yield [z]
        return
    for c in all_partitions(z, ns[0]):
        for d in all_partitions_n(c[1], ns[1:]):
            yield (c[0:1] + d)