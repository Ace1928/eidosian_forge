from typing import Iterable, Union
import numpy as np
import pennylane as qml
def create_initial_state(wires: Union[qml.wires.Wires, Iterable], prep_operation: qml.operation.StatePrepBase=None, like: str=None):
    """
    Returns an initial state, defaulting to :math:`\\ket{0}` if no state-prep operator is provided.

    Args:
        wires (Union[Wires, Iterable]): The wires to be present in the initial state
        prep_operation (Optional[StatePrepBase]): An operation to prepare the initial state
        like (Optional[str]): The machine learning interface used to create the initial state.
            Defaults to None

    Returns:
        array: The initial state of a circuit
    """
    if not prep_operation:
        num_wires = len(wires)
        state = np.zeros((2,) * num_wires)
        state[(0,) * num_wires] = 1
        return qml.math.asarray(state, like=like)
    return qml.math.asarray(prep_operation.state_vector(wire_order=list(wires)), like=like)