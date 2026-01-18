import os
from typing import Generator, Callable, Union, Sequence, Optional
from copy import copy
import warnings
import pennylane as qml
from pennylane import Snapshot
from pennylane.operation import Tensor, StatePrepBase
from pennylane.measurements import (
from pennylane.typing import ResultBatch, Result
from pennylane import DeviceError
from pennylane import transform
from pennylane.wires import WireError
@transform
def no_sampling(tape: qml.tape.QuantumTape, name: str='device') -> (Sequence[qml.tape.QuantumTape], Callable):
    """Raises an error if the tape has finite shots.

    Args:
        tape (QuantumTape or .QNode or Callable): a quantum circuit
        name (str): name to use in error message.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[.QuantumTape], function]:

        The unaltered input circuit. The output type is explained in :func:`qml.transform <pennylane.transform>`.


    This transform can be added to forbid finite shots. For example, ``default.qubit`` uses it for
    adjoint and backprop validation.
    """
    if tape.shots:
        raise qml.DeviceError(f'Finite shots are not supported with {name}')
    return ((tape,), null_postprocessing)