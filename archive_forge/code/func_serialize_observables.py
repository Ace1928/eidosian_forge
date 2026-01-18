from typing import List, Tuple
import numpy as np
from pennylane import (
from pennylane.operation import Tensor
from pennylane.tape import QuantumTape
from pennylane.math import unwrap
from pennylane import matrix, DeviceError
def serialize_observables(self, tape: QuantumTape, wires_map: dict) -> List:
    """Serializes the observables of an input tape.

        Args:
            tape (QuantumTape): the input quantum tape
            wires_map (dict): a dictionary mapping input wires to the device's backend wires

        Returns:
            list(ObsStructC128 or ObsStructC64): A list of observable objects compatible with
                the C++ backend
        """
    serialized_obs = []
    offset_indices = [0]
    for observable in tape.observables:
        ser_ob = self._ob(observable, wires_map)
        if isinstance(ser_ob, list):
            serialized_obs.extend(ser_ob)
            offset_indices.append(offset_indices[-1] + len(ser_ob))
        else:
            serialized_obs.append(ser_ob)
            offset_indices.append(offset_indices[-1] + 1)
    return (serialized_obs, offset_indices)