import sys
import numpy as np
from matplotlib import _api
@staticmethod
def get_from_args_and_kwargs(*args, **kwargs):
    """
        Return a Triangulation object from the args and kwargs, and
        the remaining args and kwargs with the consumed values removed.

        There are two alternatives: either the first argument is a
        Triangulation object, in which case it is returned, or the args
        and kwargs are sufficient to create a new Triangulation to
        return.  In the latter case, see Triangulation.__init__ for
        the possible args and kwargs.
        """
    if isinstance(args[0], Triangulation):
        triangulation, *args = args
        if 'triangles' in kwargs:
            _api.warn_external("Passing the keyword 'triangles' has no effect when also passing a Triangulation")
        if 'mask' in kwargs:
            _api.warn_external("Passing the keyword 'mask' has no effect when also passing a Triangulation")
    else:
        x, y, triangles, mask, args, kwargs = Triangulation._extract_triangulation_params(args, kwargs)
        triangulation = Triangulation(x, y, triangles, mask)
    return (triangulation, args, kwargs)