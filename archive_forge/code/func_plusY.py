import logging
import re
from dataclasses import dataclass
from typing import Any, FrozenSet, Generator, Iterable, List, Optional, cast
from pyquil.paulis import PauliTerm, sI
def plusY(q: int) -> TensorProductState:
    return TensorProductState([_OneQState(label='Y', index=0, qubit=q)])