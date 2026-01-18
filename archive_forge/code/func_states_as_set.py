import logging
import re
from dataclasses import dataclass
from typing import Any, FrozenSet, Generator, Iterable, List, Optional, cast
from pyquil.paulis import PauliTerm, sI
def states_as_set(self) -> FrozenSet[_OneQState]:
    return frozenset(self.states)