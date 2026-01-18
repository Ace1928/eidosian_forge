from functools import wraps
import inspect
from typing import Union, Callable, Tuple
import pennylane as qml
from .qnode import QNode, _make_execution_config, _get_device_shots
def expand_fn_transform(expand_fn: Callable) -> 'qml.transforms.core.TransformDispatcher':
    """Construct a transform from a tape-to-tape function.

    Args:
        expand_fn (Callable): a function from a single tape to a single tape

    Returns:

        .TransformDispatcher: Returns a transform dispatcher object that that can transform any
        circuit-like object in PennyLane.

    >>> device = qml.device('default.qubit.legacy', wires=2)
    >>> my_transform = qml.transforms.core.expand_fn_transform(device.expand_fn)
    >>> my_transform
    <transform: expand_fn>
    """

    @wraps(expand_fn)
    def wrapped_expand_fn(tape, *args, **kwargs):
        return ((expand_fn(tape, *args, **kwargs),), null_postprocessing)
    return qml.transform(wrapped_expand_fn)