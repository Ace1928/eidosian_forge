import logging
from typing import Tuple, Callable
import autograd
from autograd.numpy.numpy_boxes import ArrayBox
import pennylane as qml
def autograd_execute(tapes: Batch, execute_fn: ExecuteFn, jpc: qml.workflow.jacobian_products.JacobianProductCalculator, device=None):
    """Execute a batch of tapes with Autograd parameters on a device.

    Args:
        tapes (Sequence[.QuantumTape]): batch of tapes to execute
        execute_fn (Callable[[Sequence[.QuantumTape]], ResultBatch]): a function that turns a batch of circuits into results
        jpc (JacobianProductCalculator): a class that can compute the vector Jacobian product (VJP)
            for the input tapes.

    Returns:
        TensorLike: A nested tuple of tape results. Each element in
        the returned tuple corresponds in order to the provided tapes.

    **Example:**

    >>> from pennylane.workflow.jacobian_products import DeviceDerivatives
    >>> from pennylane.workflow.autograd import autograd_execute
    >>> execute_fn = qml.device('default.qubit').execute
    >>> config = qml.devices.ExecutionConfig(gradient_method="adjoint", use_device_gradient=True)
    >>> jpc = DeviceDerivatives(qml.device('default.qubit'), config)
    >>> def f(x):
    ...     tape = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.Z(0))])
    ...     batch = (tape, )
    ...     return autograd_execute(batch, execute_fn, jpc)
    >>> qml.grad(f)(qml.numpy.array(0.1))
    -0.09983341664682815

    """
    tapes = tuple(tapes)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug('Entry with (tapes=%s, execute_fn=%s, jpc=%s)', tapes, execute_fn, jpc)
    for tape in tapes:
        params = tape.get_parameters(trainable_only=False)
        tape.trainable_params = qml.math.get_trainable_indices(params)
    parameters = autograd.builtins.tuple([autograd.builtins.list(t.get_parameters()) for t in tapes])
    return _execute(parameters, tuple(tapes), execute_fn, jpc)