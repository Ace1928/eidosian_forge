from __future__ import annotations
import abc
import itertools
import json
import os
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MontyDecoder, MSONable
from scipy.special import factorial
@property
def ref_separation_perm(self):
    """
        Ordered indices of the separation plane.

        Examples:
            For a separation plane of type 2|4|3, with plane_points indices [0, 3, 5, 8] and
            point_groups indices [1, 4] and [2, 7, 6], the list of ordered indices is :
            [0, 3, 5, 8, 1, 4, 2, 7, 6].

        Returns:
            list[int]: of ordered indices of this separation plane.
        """
    return self._ref_separation_perm