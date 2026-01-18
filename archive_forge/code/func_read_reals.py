import warnings
import numpy as np
def read_reals(self, dtype='f8'):
    """
        Reads a record of a given type from the file, defaulting to a floating
        point number (``real*8`` in Fortran).

        Parameters
        ----------
        dtype : dtype, optional
            Data type specifying the size and endianness of the data.

        Returns
        -------
        data : ndarray
            A 1-D array object.

        See Also
        --------
        read_ints
        read_record

        """
    return self.read_record(dtype)