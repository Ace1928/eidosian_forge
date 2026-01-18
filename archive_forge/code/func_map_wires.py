from functools import wraps
from typing import Type
from pennylane import QueuingManager
from pennylane.operation import AnyWires, Operation, Operator
from pennylane.tape import make_qscript
from pennylane.compiler import compiler
def map_wires(self, wire_map):
    meas_val = self.meas_val.map_wires(wire_map)
    then_op = self.then_op.map_wires(wire_map)
    return Conditional(meas_val, then_op=then_op)