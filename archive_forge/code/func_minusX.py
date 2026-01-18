import logging
import re
from dataclasses import dataclass
from typing import Any, FrozenSet, Generator, Iterable, List, Optional, cast
from pyquil.paulis import PauliTerm, sI
def minusX(q: int) -> TensorProductState:
    return TensorProductState([_OneQState(label='X', index=1, qubit=q)])