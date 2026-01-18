import numpy as np
import pennylane as qml
from pennylane.measurements import ProbabilityMP
def batch_jvp(tapes, tangents, gradient_fn, reduction='append', gradient_kwargs=None):
    """Generate the gradient tapes and processing function required to compute
    the Jacobian vector products of a batch of tapes.

    Args:
        tapes (Sequence[.QuantumTape]): sequence of quantum tapes to differentiate
        tangents (Sequence[tensor_like]): Sequence of gradient-output vectors ``dy``. Must be the
            same length as ``tapes``. Each ``dy`` tensor should have shape
            matching the output shape of the corresponding tape.
        gradient_fn (callable): the gradient transform to use to differentiate
            the tapes
        reduction (str): Determines how the Jacobian-vector products are returned.
            If ``append``, then the output of the function will be of the form
            ``List[tensor_like]``, with each element corresponding to the JVP of each
            input tape. If ``extend``, then the output JVPs will be concatenated.
        gradient_kwargs (dict): dictionary of keyword arguments to pass when
            determining the gradients of tapes

    Returns:
        List[tensor_like or None]: list of Jacobian vector products. ``None`` elements corresponds
        to tapes with no trainable parameters.

    **Example**

    .. code-block:: python

        import jax
        x = jax.numpy.array([[0.1, 0.2, 0.3],
                             [0.4, 0.5, 0.6]])

        ops = [
            qml.RX(x[0, 0], wires=0),
            qml.RY(x[0, 1], wires=1),
            qml.RZ(x[0, 2], wires=0),
            qml.CNOT(wires=[0, 1]),
            qml.RX(x[1, 0], wires=1),
            qml.RY(x[1, 1], wires=0),
            qml.RZ(x[1, 2], wires=1)
        ]
        measurements1 = [qml.expval(qml.Z(0)), qml.probs(wires=1)]
        tape1 = qml.tape.QuantumTape(ops, measurements1)

        measurements2 = [qml.expval(qml.Z(0) @ qml.Z(1))]
        tape2 = qml.tape.QuantumTape(ops, measurements2)

        tapes = [tape1, tape2]

    Both tapes share the same circuit ansatz, but have different measurement outputs.

    We can use the ``batch_jvp`` function to compute the Jacobian vector product,
    given a list of tangents ``tangent``:

    >>> tangent_0 = [jax.numpy.array(1.0), jax.numpy.array(1.0), jax.numpy.array(1.0), jax.numpy.array(1.0), jax.numpy.array(1.0), jax.numpy.array(1.0)]
    >>> tangent_1 = [jax.numpy.array(1.0), jax.numpy.array(1.0), jax.numpy.array(1.0), jax.numpy.array(1.0), jax.numpy.array(1.0), jax.numpy.array(1.0)]
    >>> tangents = [tangent_0, tangent_1]

    Note that each ``tangents`` has shape matching the parameter dimension of the tape.

    Executing the JVP tapes, and applying the processing function:

    >>> jvp_tapes, fn = qml.gradients.batch_jvp(tapes, tangents, qml.gradients.param_shift)

    >>> dev = qml.device("default.qubit", wires=2)
    >>> jvps = fn(dev.execute(jvp_tapes))
    >>> jvps
    ((Array(-0.62073976, dtype=float32), Array([-0.3259707 ,  0.32597077], dtype=float32)), Array(-0.6900841, dtype=float32))

    We have two JVPs; one per tape. Each one corresponds to the shape of the output of their respective tape.
    """
    gradient_kwargs = gradient_kwargs or {}
    reshape_info = []
    gradient_tapes = []
    processing_fns = []
    for tape, tangent in zip(tapes, tangents):
        g_tapes, fn = jvp(tape, tangent, gradient_fn, gradient_kwargs)
        reshape_info.append(len(g_tapes))
        processing_fns.append(fn)
        gradient_tapes.extend(g_tapes)

    def processing_fn(results):
        jvps = []
        start = 0
        for t_idx in range(len(tapes)):
            res_len = reshape_info[t_idx]
            res_t = results[start:start + res_len]
            start += res_len
            jvp_ = processing_fns[t_idx](res_t)
            if jvp_ is None:
                if reduction == 'append':
                    jvps.append(None)
                continue
            if isinstance(reduction, str):
                getattr(jvps, reduction)(jvp_)
            elif callable(reduction):
                reduction(jvps, jvp_)
        return tuple(jvps)
    return (gradient_tapes, processing_fn)