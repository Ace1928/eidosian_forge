from functools import partial
from typing import Callable, Sequence
import pennylane as qml
from pennylane.tape import QuantumTape
from pennylane.devices import DefaultQubit, DefaultQubitLegacy, DefaultMixed
from pennylane.measurements import StateMP, DensityMatrixMP
from pennylane.gradients import adjoint_metric_tensor, metric_tensor
from pennylane import transform
def evaluate_trace_distance(all_args0=None, all_args1=None):
    """Wrapper used for evaluation of the trace distance between two states computed from
        QNodes. It allows giving the args and kwargs to each :class:`.QNode`.

        Args:
            all_args0 (tuple): Tuple containing the arguments (*args, kwargs) of the first :class:`.QNode`.
            all_args1 (tuple): Tuple containing the arguments (*args, kwargs) of the second :class:`.QNode`.

        Returns:
            float: Trace distance between two quantum states
        """
    if not isinstance(all_args0, tuple) and all_args0 is not None:
        all_args0 = (all_args0,)
    if not isinstance(all_args1, tuple) and all_args1 is not None:
        all_args1 = (all_args1,)
    if all_args0 is not None:
        if isinstance(all_args0[-1], dict):
            args0 = all_args0[:-1]
            kwargs0 = all_args0[-1]
        else:
            args0 = all_args0
            kwargs0 = {}
        state0 = state_qnode0(*args0, **kwargs0)
    else:
        state0 = state_qnode0()
    if all_args1 is not None:
        if isinstance(all_args1[-1], dict):
            args1 = all_args1[:-1]
            kwargs1 = all_args1[-1]
        else:
            args1 = all_args1
            kwargs1 = {}
        state1 = state_qnode1(*args1, **kwargs1)
    else:
        state1 = state_qnode1()
    return qml.math.trace_distance(state0, state1)