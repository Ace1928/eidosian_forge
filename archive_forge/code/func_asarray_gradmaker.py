import numpy as onp
from autograd import numpy as _np
from autograd.core import VSpace
from autograd.extend import defvjp, primitive
from autograd.numpy.numpy_boxes import ArrayBox
from autograd.numpy.numpy_vspaces import ArrayVSpace, ComplexArrayVSpace
from autograd.tracer import Box
from pennylane.operation import Operator
def asarray_gradmaker(ans, *args, **kwargs):
    """Gradient maker for asarray"""
    del ans, args, kwargs
    return lambda g: g