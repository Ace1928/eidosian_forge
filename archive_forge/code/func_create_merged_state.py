import abc
from typing import (
from typing_extensions import Self
import numpy as np
from cirq import protocols, value
from cirq.type_workarounds import NotImplementedType
@abc.abstractmethod
def create_merged_state(self) -> TSimulationState:
    """Creates a final merged state."""