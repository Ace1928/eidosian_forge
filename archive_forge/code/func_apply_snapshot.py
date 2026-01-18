from functools import singledispatch
from string import ascii_letters as alphabet
import numpy as np
import pennylane as qml
from pennylane import math
from pennylane.measurements import MidMeasureMP
from pennylane.ops import Conditional
@apply_operation.register
def apply_snapshot(op: qml.Snapshot, state, is_state_batched: bool=False, debugger=None, **_):
    """Take a snapshot of the state"""
    if debugger is not None and debugger.active:
        measurement = op.hyperparameters['measurement']
        if measurement:
            snapshot = qml.devices.qubit.measure(measurement, state)
        else:
            flat_shape = (math.shape(state)[0], -1) if is_state_batched else (-1,)
            snapshot = math.cast(math.reshape(state, flat_shape), complex)
        if op.tag:
            debugger.snapshots[op.tag] = snapshot
        else:
            debugger.snapshots[len(debugger.snapshots)] = snapshot
    return state