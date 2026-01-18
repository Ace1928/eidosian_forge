import operator
import warnings
def canonicalize_axis(axis, num_dims):
    """Canonicalize an axis in [-num_dims, num_dims) to [0, num_dims)."""
    axis = operator.index(axis)
    if not -num_dims <= axis < num_dims:
        raise ValueError(f'axis {axis} is out of bounds for an array with dimension {num_dims}.')
    if axis < 0:
        axis = axis + num_dims
    return axis