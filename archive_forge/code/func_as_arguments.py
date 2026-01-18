from collections import deque
from numba.core import types, cgutils
def as_arguments(self, builder, values):
    """Flatten all argument values
        """
    if len(values) != self._nargs:
        raise TypeError('invalid number of args: expected %d, got %d' % (self._nargs, len(values)))
    if not values:
        return ()
    args = [dm.as_argument(builder, val) for dm, val in zip(self._dm_args, values)]
    args = tuple(_flatten(args))
    return args