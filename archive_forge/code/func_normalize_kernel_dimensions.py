import numbers
from numba.core.errors import LoweringError
def normalize_kernel_dimensions(griddim, blockdim):
    """
    Normalize and validate the user-supplied kernel dimensions.
    """

    def check_dim(dim, name):
        if not isinstance(dim, (tuple, list)):
            dim = [dim]
        else:
            dim = list(dim)
        if len(dim) > 3:
            raise ValueError('%s must be a sequence of 1, 2 or 3 integers, got %r' % (name, dim))
        for v in dim:
            if not isinstance(v, numbers.Integral):
                raise TypeError('%s must be a sequence of integers, got %r' % (name, dim))
        while len(dim) < 3:
            dim.append(1)
        return tuple(dim)
    if None in (griddim, blockdim):
        raise ValueError(missing_launch_config_msg)
    griddim = check_dim(griddim, 'griddim')
    blockdim = check_dim(blockdim, 'blockdim')
    return (griddim, blockdim)