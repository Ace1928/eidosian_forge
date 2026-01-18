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
def permutations(self):
    """
        Permutations used for this separation plane algorithm.

        Returns:
            list[Permutations]: to be performed.
        """
    return self._permutations