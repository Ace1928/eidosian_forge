import numpy as np
from .casting import best_float, floor_exact, int_abs, shared_range, type_info
from .volumeutils import array_to_file, finite_range
def scaling_needed(self):
    """Checks if scaling is needed for input array

        Raises WriterError if no scaling possible.

        The rules are in the code, but:

        * If numpy will cast, return False (no scaling needed)
        * If input or output is an object or structured type, raise
        * If input is complex, raise
        * If the output is float, return False
        * If the input array is all zero, return False
        * If there is no finite value, return False (the writer will strip the
          non-finite values)
        * By now we are casting to (u)int. If the input type is a float, return
          True (we do need scaling)
        * Now input and output types are (u)ints. If the min and max in the
          data are within range of the output type, return False
        * Otherwise return True
        """
    if not super().scaling_needed():
        return False
    mn, mx = self.finite_range()
    return (mn, mx) != (np.inf, -np.inf)