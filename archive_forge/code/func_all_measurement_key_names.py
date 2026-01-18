from typing import (
import numpy as np
from cirq import protocols, _compat
from cirq.circuits import AbstractCircuit, Alignment, Circuit
from cirq.circuits.insert_strategy import InsertStrategy
from cirq.type_workarounds import NotImplementedType
@_compat.cached_method
def all_measurement_key_names(self) -> FrozenSet[str]:
    return frozenset((str(key) for key in self.all_measurement_key_objs()))