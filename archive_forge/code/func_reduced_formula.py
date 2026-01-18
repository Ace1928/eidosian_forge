from __future__ import annotations
from abc import ABC, ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Literal
import numpy as np
from monty.json import MSONable
from pymatgen.core.composition import Composition
@property
def reduced_formula(self) -> str:
    """The reduced formula of the entry."""
    return self._composition.reduced_formula