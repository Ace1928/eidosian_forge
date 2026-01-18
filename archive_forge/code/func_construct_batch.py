from functools import wraps
import inspect
from typing import Union, Callable, Tuple
import pennylane as qml
from .qnode import QNode, _make_execution_config, _get_device_shots
def construct_batch(qnode: QNode, level: Union[None, str, int, slice]='user') -> Callable:
    """Construct the batch of tapes and post processing for a designated stage in the transform program.

    Args:
        qnode (QNode): the qnode we want to get the tapes and post-processing for.
        level (None, str, int, slice): And indication of what transforms to use from the full program.

            * ``None``: use the full transform program
            * ``str``: Acceptable keys are ``"top"``, ``"user"``, ``"device"``, and ``"gradient"``
            * ``int``: How many transforms to include, starting from the front of the program
            * ``slice``: a slice to select out components of the transform program.

    Returns:
        Callable:  a function with the same call signature as the initial quantum function. This function returns
        a batch (tuple) of tapes and postprocessing function.

    .. seealso:: :func:`pennylane.workflow.get_transform_program` to inspect the contents of the transform program for a specified level.


    .. details::
        :title: Usage Details

        Suppose we have a QNode with several user transforms.

        .. code-block:: python

            @qml.transforms.undo_swaps
            @qml.transforms.merge_rotations
            @qml.transforms.cancel_inverses
            @qml.qnode(qml.device('default.qubit'), diff_method="parameter-shift", shifts=np.pi / 4)
            def circuit(x):
                qml.RandomLayers(qml.numpy.array([[1.0, 2.0]]), wires=(0,1))
                qml.RX(x, wires=0)
                qml.RX(-x, wires=0)
                qml.SWAP((0,1))
                qml.X(0)
                qml.X(0)
                return qml.expval(qml.X(0) + qml.Y(0))

        We can inspect what the device will execute with:

        >>> batch, fn = construct_batch(circuit, level="device")(1.23)
        >>> batch[0].circuit
        [RY(tensor(1., requires_grad=True), wires=[1]),
        RX(tensor(2., requires_grad=True), wires=[0]),
        expval(  (1) [X0]
        + (1) [Y0])]

        These tapes can be natively executed by the device, though with non-backprop devices the parameters
        will need to be converted to numpy with :func:`~.convert_to_numpy_parameters`.

        >>> fn(dev.execute(batch))
        (tensor(-0.90929743, requires_grad=True),)

        Or what the parameter shift gradient transform will be applied to:

        >>> batch, fn = construct_batch(circuit, level="gradient")(1.23)
        >>> batch[0].circuit
        [RY(tensor(1., requires_grad=True), wires=[1]),
        RX(tensor(2., requires_grad=True), wires=[0]),
        expval(  (1) [X0]
        + (1) [Y0])]

        We can inspect what was directly captured from the qfunc with ``level=0``.

        >>> batch, fn = construct_batch(circuit, level=0)(1.23)
        >>> batch[0].circuit
        [RandomLayers(tensor([[1., 2.]], requires_grad=True), wires=[0, 1]),
        RX(1.23, wires=[0]),
        RX(-1.23, wires=[0]),
        SWAP(wires=[0, 1]),
        X(0),
        X(0),
        expval(  (1) [X0]
        + (1) [Y0])]

        And iterate though stages in the transform program with different integers.
        If we request ``level=1``, the ``cancel_inverses`` transform has been applied.

        >>> batch, fn = construct_batch(circuit, level=1)(1.23)
        >>> batch[0].circuit
        [RandomLayers(tensor([[1., 2.]], requires_grad=True), wires=[0, 1]),
        RX(1.23, wires=[0]),
        RX(-1.23, wires=[0]),
        SWAP(wires=[0, 1]),
        expval(  (1) [X0]
        + (1) [Y0])]

        We can also slice into a subset of the transform program.  ``slice(1, None)`` would skip the first user
        transform ``cancel_inverses``:

        >>> batch, fn = construct_batch(circuit, level=slice(1,None))(1.23)
        >>> batch[0].circuit
        [RY(tensor(1., requires_grad=True), wires=[1]),
        RX(tensor(2., requires_grad=True), wires=[0]),
        X(0),
        X(0),
        expval(  (1) [X0]
        + (1) [Y0])]

    """
    program = get_transform_program(qnode, level=level)

    def batch_constructor(*args, **kwargs) -> Tuple[Tuple['qml.tape.QuantumTape', Callable]]:
        """Create a batch of tapes and a post processing function."""
        if 'shots' in inspect.signature(qnode.func).parameters:
            shots = _get_device_shots(qnode.device)
        else:
            shots = kwargs.pop('shots', _get_device_shots(qnode.device))
        initial_tape = qml.tape.make_qscript(qnode.func, shots=shots)(*args, **kwargs)
        return program((initial_tape,))
    return batch_constructor