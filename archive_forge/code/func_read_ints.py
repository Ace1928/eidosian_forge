import warnings
import numpy as np
def read_ints(self, dtype='i4'):
    """
        Reads a record of a given type from the file, defaulting to an integer
        type (``INTEGER*4`` in Fortran).

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
        read_reals
        read_record

        """
    return self.read_record(dtype)