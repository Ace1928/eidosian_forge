from typing import List, Tuple
import numpy as np
from pennylane import (
from pennylane.operation import Tensor
from pennylane.tape import QuantumTape
from pennylane.math import unwrap
from pennylane import matrix, DeviceError
def serialize_ops(self, tape: QuantumTape, wires_map: dict) -> Tuple[List[List[str]], List[np.ndarray], List[List[int]], List[bool], List[np.ndarray], List[List[int]], List[List[bool]]]:
    """Serializes the operations of an input tape.

        The state preparation operations are not included.

        Args:
            tape (QuantumTape): the input quantum tape
            wires_map (dict): a dictionary mapping input wires to the device's backend wires

        Returns:
            Tuple[list, list, list, list, list]: A serialization of the operations, containing a
            list of operation names, a list of operation parameters, a list of observable wires,
            a list of inverses, and a list of matrices for the operations that do not have a
            dedicated kernel.
        """
    names = []
    params = []
    controlled_wires = []
    controlled_values = []
    wires = []
    mats = []
    uses_stateprep = False

    def get_wires(operation, single_op):
        if operation.name[0:2] == 'C(' or operation.name == 'MultiControlledX':
            name = 'PauliX' if operation.name == 'MultiControlledX' else operation.base.name
            controlled_wires_list = operation.control_wires
            if operation.name == 'MultiControlledX':
                wires_list = list(set(operation.wires) - set(controlled_wires_list))
            else:
                wires_list = operation.target_wires
            control_values_list = [bool(int(i)) for i in operation.hyperparameters['control_values']] if operation.name == 'MultiControlledX' else operation.control_values
            if not hasattr(self.sv_type, name):
                single_op = QubitUnitary(matrix(single_op.base), single_op.base.wires)
                name = single_op.name
        else:
            name = single_op.name
            wires_list = single_op.wires.tolist()
            controlled_wires_list = []
            control_values_list = []
        return (single_op, name, wires_list, controlled_wires_list, control_values_list)
    for operation in tape.operations:
        if isinstance(operation, (BasisState, StatePrep)):
            uses_stateprep = True
            continue
        if isinstance(operation, Rot):
            op_list = operation.expand().operations
        else:
            op_list = [operation]
        for single_op in op_list:
            single_op, name, wires_list, controlled_wires_list, controlled_values_list = get_wires(operation, single_op)
            names.append(name)
            if name == 'QubitUnitary':
                params.append([0.0])
                mats.append(matrix(single_op))
            elif not hasattr(self.sv_type, name):
                params.append([])
                mats.append(matrix(single_op))
            else:
                params.append(single_op.parameters)
                mats.append([])
            controlled_values.append(controlled_values_list)
            controlled_wires.append([wires_map[w] for w in controlled_wires_list])
            wires.append([wires_map[w] for w in wires_list])
    inverses = [False] * len(names)
    return ((names, params, wires, inverses, mats, controlled_wires, controlled_values), uses_stateprep)