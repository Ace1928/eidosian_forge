from __future__ import annotations
import os
import re
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from fractions import Fraction
from itertools import product
from typing import TYPE_CHECKING, ClassVar, Literal, overload
import numpy as np
from monty.design_patterns import cached_class
from monty.serialization import loadfn
from pymatgen.util.string import Stringify
def is_supergroup(self, subgroup: SymmetryGroup) -> bool:
    """True if this space group is a supergroup of the supplied group.

        Args:
            subgroup (Spacegroup): Subgroup to test.

        Returns:
            bool: True if this space group is a supergroup of the supplied group.
        """
    return subgroup.is_subgroup(self)