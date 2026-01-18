import contextlib
import copy
from collections import Counter
from typing import List, Union, Optional, Sequence
import pennylane as qml
from pennylane.measurements import (
from pennylane.typing import TensorLike
from pennylane.operation import Observable, Operator, Operation, _UNSET_BATCH_SIZE
from pennylane.pytrees import register_pytree
from pennylane.queuing import AnnotatedQueue, process_queue
from pennylane.wires import Wires
def map_to_standard_wires(self):
    """
        Map a circuit's wires such that they are in a standard order. If no
        mapping is required, the unmodified circuit is returned.

        Returns:
            QuantumScript: The circuit with wires in the standard order

        The standard order is defined by the operator wires being increasing
        integers starting at zero, to match array indices. If there are any
        measurement wires that are not in any operations, those will be mapped
        to higher values.

        **Example:**

        >>> circuit = qml.tape.QuantumScript([qml.X("a")], [qml.expval(qml.Z("b"))])
        >>> circuit.map_to_standard_wires().circuit
        [X(0), expval(Z(1))]

        If any measured wires are not in any operations, they will be mapped last:

        >>> circuit = qml.tape.QuantumScript([qml.X(1)], [qml.probs(wires=[0, 1])])
        >>> circuit.map_to_standard_wires().circuit
        [X(0), probs(wires=[1, 0])]

        If no wire-mapping is needed, then the returned circuit *is* the inputted circuit:

        >>> circuit = qml.tape.QuantumScript([qml.X(0)], [qml.expval(qml.Z(1))])
        >>> circuit.map_to_standard_wires() is circuit
        True

        """
    op_wires = Wires.all_wires((op.wires for op in self.operations))
    meas_wires = Wires.all_wires((mp.wires for mp in self.measurements))
    num_op_wires = len(op_wires)
    meas_only_wires = set(meas_wires) - set(op_wires)
    if set(op_wires) == set(range(num_op_wires)) and meas_only_wires == set(range(num_op_wires, num_op_wires + len(meas_only_wires))):
        return self
    wire_map = {w: i for i, w in enumerate(op_wires + meas_only_wires)}
    tapes, fn = qml.map_wires(self, wire_map)
    return fn(tapes)