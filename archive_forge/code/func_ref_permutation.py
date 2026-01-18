from __future__ import annotations
import abc
import itertools
import json
import os
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MontyDecoder, MSONable
from scipy.special import factorial
def ref_permutation(self, permutation):
    """
        Returns the reference permutation for a set of equivalent permutations.

        Can be useful to skip permutations that have already been performed.

        Args:
            permutation: Current permutation

        Returns:
            Permutation: Reference permutation of the perfect CoordinationGeometry.
        """
    perms = []
    for eqv_indices in self.equivalent_indices:
        perms.append(tuple((permutation[ii] for ii in eqv_indices)))
    perms.sort()
    return perms[0]