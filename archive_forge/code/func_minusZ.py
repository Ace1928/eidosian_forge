import logging
import re
from dataclasses import dataclass
from typing import Any, FrozenSet, Generator, Iterable, List, Optional, cast
from pyquil.paulis import PauliTerm, sI
def minusZ(q: int) -> TensorProductState:
    return TensorProductState([_OneQState(label='Z', index=1, qubit=q)])