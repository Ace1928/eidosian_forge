from copy import copy
from typing import Tuple
import numpy as np
import numpy.linalg as npl
import pennylane as qml
from pennylane.operation import Operation, Operator
from pennylane.wires import Wires
from pennylane import math
def get_single_qubit_rot_angles_via_matrix() -> Tuple[float, float, float]:
    """Returns a triplet of angles representing the single-qubit decomposition
        of the matrix of the target operation using ZYZ rotations.
        """
    with qml.QueuingManager.stop_recording():
        zyz_decomp = qml.ops.one_qubit_decomposition(qml.matrix(target_operation), wire=target_wire, rotations='ZYZ')
    return tuple((gate.parameters[0] for gate in zyz_decomp))