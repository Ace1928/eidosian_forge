import functools
from collections.abc import Sequence
import autoray as ar
import numpy as onp
from autograd.numpy.numpy_boxes import ArrayBox
from autoray import numpy as np
from numpy import ndarray
from . import single_dispatch  # pylint:disable=unused-import
from .utils import cast, cast_like, get_interface, requires_grad
def jax_argnums_to_tape_trainable(qnode, argnums, program, args, kwargs):
    """This functions gets the tape parameters from the QNode construction given some argnums (only for Jax).
    The tape parameters are transformed to JVPTracer if they are from argnums. This function imitates the behavior
    of Jax in order to mark trainable parameters.

    Args:
        qnode(qml.QNode): the quantum node.
        argnums(int, list[int]): the parameters that we want to set as trainable (on the QNode level).
        program(qml.transforms.core.TransformProgram): the transform program to be applied on the tape.


    Return:
        list[float, jax.JVPTracer]: List of parameters where the trainable one are `JVPTracer`.
    """
    import jax
    with jax.core.new_main(jax.interpreters.ad.JVPTrace) as main:
        trace = jax.interpreters.ad.JVPTrace(main, 0)
    args_jvp = [jax.interpreters.ad.JVPTracer(trace, arg, jax.numpy.zeros(arg.shape)) if i in argnums else arg for i, arg in enumerate(args)]
    qnode.construct(args_jvp, kwargs)
    tape = qnode.qtape
    tapes, _ = program((tape,))
    del trace
    return tuple((tape.get_parameters(trainable_only=False) for tape in tapes))