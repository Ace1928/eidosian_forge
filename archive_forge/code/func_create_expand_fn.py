import contextlib
import pennylane as qml
from pennylane.operation import (
def create_expand_fn(depth, stop_at=None, device=None, docstring=None):
    """Create a function for expanding a tape to a given depth, and
    with a specific stopping criterion. This is a wrapper around
    :meth:`~.QuantumTape.expand`.

    Args:
        depth (int): Depth for the expansion
        stop_at (callable): Stopping criterion. This must be a function with signature
            ``stop_at(obj)``, where ``obj`` is a *queueable* PennyLane object such as
            :class:`~.Operation` or :class:`~.MeasurementProcess`. It must return a
            boolean, indicating if the expansion should stop at this object.
        device (pennylane.Device): Ensure that the expanded tape only uses native gates of the
            given device.
        docstring (str): docstring for the generated expansion function

    Returns:
        callable: Tape expansion function. The returned function accepts a :class:`~.QuantumTape`,
        and returns an expanded :class:`~.QuantumTape`.

    **Example**

    Let us construct an expansion function that expands a tape in order to
    decompose trainable multi-parameter gates. We allow for up to five expansion
    steps, which can be controlled with the argument ``depth``.
    The stopping criterion is easy to write as

    >>> stop_at = ~(qml.operation.has_multipar & qml.operation.is_trainable)

    Then the expansion function can be obtained via

    >>> expand_fn = qml.transforms.create_expand_fn(depth=5, stop_at=stop_at)

    We can test the newly generated function on an example tape:

    .. code-block:: python

        ops = [
            qml.RX(0.2, wires=0),
            qml.RX(qml.numpy.array(-2.4, requires_grad=True), wires=1),
            qml.Rot(1.7, 0.92, -1.1, wires=0),
            qml.Rot(*qml.numpy.array([-3.1, 0.73, 1.36], requires_grad=True), wires=1)
        ]
        tape = qml.tape.QuantumTape(ops)

    >>> new_tape = expand_fn(tape)
    >>> print(qml.drawer.tape_text(tape, decimals=1))
    0: ──RX(0.2)───Rot(1.7,0.9,-1.1)─┤
    1: ──RX(-2.4)──Rot(-3.1,0.7,1.4)─┤
    >>> print(qml.drawer.tape_text(new_tape, decimals=1))
    0: ──RX(0.2)───Rot(1.7,0.9,-1.1)───────────────────┤
    1: ──RX(-2.4)──RZ(-3.1)───────────RY(0.7)──RZ(1.4)─┤

    """
    if device is not None:
        if stop_at is None:
            stop_at = device.stopping_condition
        else:
            stop_at &= device.stopping_condition

    def expand_fn(tape, depth=depth, **kwargs):
        with qml.QueuingManager.stop_recording():
            if stop_at is None:
                tape = tape.expand(depth=depth)
            elif not all((stop_at(op) for op in tape.operations)):
                tape = tape.expand(depth=depth, stop_at=stop_at)
            else:
                return tape
            _update_trainable_params(tape)
        return tape
    if docstring:
        expand_fn.__doc__ = docstring
    return expand_fn