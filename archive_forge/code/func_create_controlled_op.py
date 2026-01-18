import warnings
import functools
from copy import copy
from functools import wraps
from inspect import signature
from typing import List
import numpy as np
from scipy import sparse
import pennylane as qml
from pennylane import operation
from pennylane import math as qmlmath
from pennylane.operation import Operator
from pennylane.wires import Wires
from pennylane.compiler import compiler
from .symbolicop import SymbolicOp
from .controlled_decompositions import ctrl_decomp_bisect, ctrl_decomp_zyz
def create_controlled_op(op, control, control_values=None, work_wires=None):
    """Default ``qml.ctrl`` implementation, allowing other implementations to call it when needed."""
    control = qml.wires.Wires(control)
    if isinstance(control_values, (int, bool)):
        control_values = [control_values]
    elif control_values is None:
        control_values = [True] * len(control)
    ctrl_op = _try_wrap_in_custom_ctrl_op(op, control, control_values=control_values, work_wires=work_wires)
    if ctrl_op is not None:
        return ctrl_op
    pauli_x_based_ctrl_ops = _get_pauli_x_based_ops()
    if isinstance(op, pauli_x_based_ctrl_ops):
        qml.QueuingManager.remove(op)
        return _handle_pauli_x_based_controlled_ops(op, control, control_values, work_wires)
    if isinstance(op, Controlled):
        work_wires = work_wires or []
        return ctrl(op.base, control=control + op.control_wires, control_values=control_values + op.control_values, work_wires=work_wires + op.work_wires)
    if isinstance(op, Operator):
        return Controlled(op, control_wires=control, control_values=control_values, work_wires=work_wires)
    if not callable(op):
        raise ValueError(f'The object {op} of type {type(op)} is not an Operator or callable. This error might occur if you apply ctrl to a list of operations instead of a function or Operator.')

    @wraps(op)
    def wrapper(*args, **kwargs):
        qscript = qml.tape.make_qscript(op)(*args, **kwargs)
        flip_control_on_zero = len(qscript) > 1 and control_values is not None
        op_control_values = None if flip_control_on_zero else control_values
        if flip_control_on_zero:
            _ = [qml.X(w) for w, val in zip(control, control_values) if not val]
        _ = [ctrl(op, control=control, control_values=op_control_values, work_wires=work_wires) for op in qscript.operations]
        if flip_control_on_zero:
            _ = [qml.X(w) for w, val in zip(control, control_values) if not val]
        if qml.QueuingManager.recording():
            _ = [qml.apply(m) for m in qscript.measurements]
        return qscript.measurements
    return wrapper