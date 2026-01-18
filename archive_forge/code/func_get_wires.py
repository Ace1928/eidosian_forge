from typing import List, Tuple
import numpy as np
from pennylane import (
from pennylane.operation import Tensor
from pennylane.tape import QuantumTape
from pennylane.math import unwrap
from pennylane import matrix, DeviceError
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