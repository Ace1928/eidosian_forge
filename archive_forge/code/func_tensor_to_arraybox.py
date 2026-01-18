import numpy as onp
from autograd import numpy as _np
from autograd.core import VSpace
from autograd.extend import defvjp, primitive
from autograd.numpy.numpy_boxes import ArrayBox
from autograd.numpy.numpy_vspaces import ArrayVSpace, ComplexArrayVSpace
from autograd.tracer import Box
from pennylane.operation import Operator
def tensor_to_arraybox(x, *args):
    """Convert a :class:`~.tensor` to an Autograd ``ArrayBox``.

    Args:
        x (array_like): Any data structure in any form that can be converted to
            an array. This includes lists, lists of tuples, tuples, tuples of tuples,
            tuples of lists and ndarrays.

    Returns:
        autograd.numpy.numpy_boxes.ArrayBox: Autograd ArrayBox instance of the array

    Raises:
        NonDifferentiableError: if the provided tensor is non-differentiable
    """
    if isinstance(x, tensor):
        if x.requires_grad:
            return ArrayBox(x, *args)
        raise NonDifferentiableError(f'{x} is non-differentiable. Set the requires_grad attribute to True.')
    return ArrayBox(x, *args)