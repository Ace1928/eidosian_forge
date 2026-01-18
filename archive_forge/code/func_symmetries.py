from __future__ import annotations
import itertools
from collections.abc import Iterable
from copy import deepcopy
from typing import Union, cast
import numpy as np
from qiskit.exceptions import QiskitError
from ..operators import Pauli, SparsePauliOp
@property
def symmetries(self) -> list[Pauli]:
    """Return symmetries."""
    return self._symmetries