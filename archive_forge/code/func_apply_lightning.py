from pathlib import Path
from typing import List, Sequence
from warnings import warn
import numpy as np
from pennylane_lightning.core.lightning_base import (
def apply_lightning(self, operations):
    """Apply a list of operations to the state tensor.

            Args:
                operations (list[~pennylane.operation.Operation]): operations to apply

            Returns:
                array[complex]: the output state tensor
            """
    state = self.state_vector
    for operation in operations:
        if isinstance(operation, Adjoint):
            name = operation.base.name
            invert_param = True
        else:
            name = operation.name
            invert_param = False
        if name == 'Identity':
            continue
        method = getattr(state, name, None)
        wires = self.wires.indices(operation.wires)
        if method is not None:
            param = operation.parameters
            method(wires, invert_param, param)
        elif name[0:2] == 'C(' or name == 'ControlledQubitUnitary' or name == 'MultiControlledX':
            self._apply_lightning_controlled(operation)
        else:
            method = getattr(state, 'applyMatrix')
            try:
                method(qml.matrix(operation), wires, False)
            except AttributeError:
                method(operation.matrix, wires, False)