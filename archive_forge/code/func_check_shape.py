import cupy
import operator
import numpy
from cupy._core._dtype import get_dtype
def check_shape(args, current_shape=None):
    """Check validity of the shape"""
    if len(args) == 0:
        raise TypeError("function missing 1 required positional argument: 'shape'")
    elif len(args) == 1:
        try:
            shape_iter = iter(args[0])
        except TypeError:
            new_shape = (operator.index(args[0]),)
        else:
            new_shape = tuple((operator.index(arg) for arg in shape_iter))
    else:
        new_shape = tuple((operator.index(arg) for arg in args))
    if current_shape is None:
        if len(new_shape) != 2:
            raise ValueError('shape must be a 2-tuple of positive integers')
        elif new_shape[0] < 0 or new_shape[1] < 0:
            raise ValueError("'shape' elements cannot be negative")
    else:
        current_size = numpy.prod(current_shape)
        negative_indexes = [i for i, x in enumerate(new_shape) if x < 0]
        if len(negative_indexes) == 0:
            new_size = numpy.prod(new_shape)
            if new_size != current_size:
                raise ValueError('cannot reshape array of size {} into shape{}'.format(current_size, new_shape))
        elif len(negative_indexes) == 1:
            skip = negative_indexes[0]
            specified = numpy.prod(new_shape[0:skip] + new_shape[skip + 1:])
            unspecified, remainder = divmod(current_size, specified)
            if remainder != 0:
                err_shape = tuple(('newshape' if x < 0 else x for x in new_shape))
                raise ValueError('cannot reshape array of size {} into shape{}'.format(current_size, err_shape))
            new_shape = new_shape[0:skip] + (unspecified,) + new_shape[skip + 1:]
        else:
            raise ValueError('can only specify one unknown dimension')
    if len(new_shape) != 2:
        raise ValueError('matrix shape must be two-dimensional')
    return new_shape