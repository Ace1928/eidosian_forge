import warnings
import numpy as np
def read_record(self, *dtypes, **kwargs):
    """
        Reads a record of a given type from the file.

        Parameters
        ----------
        *dtypes : dtypes, optional
            Data type(s) specifying the size and endianness of the data.

        Returns
        -------
        data : ndarray
            A 1-D array object.

        Raises
        ------
        FortranEOFError
            To signal that no further records are available
        FortranFormattingError
            To signal that the end of the file was encountered
            part-way through a record

        Notes
        -----
        If the record contains a multidimensional array, you can specify
        the size in the dtype. For example::

            INTEGER var(5,4)

        can be read with::

            read_record('(4,5)i4').T

        Note that this function does **not** assume the file data is in Fortran
        column major order, so you need to (i) swap the order of dimensions
        when reading and (ii) transpose the resulting array.

        Alternatively, you can read the data as a 1-D array and handle the
        ordering yourself. For example::

            read_record('i4').reshape(5, 4, order='F')

        For records that contain several variables or mixed types (as opposed
        to single scalar or array types), give them as separate arguments::

            double precision :: a
            integer :: b
            write(1) a, b

            record = f.read_record('<f4', '<i4')
            a = record[0]  # first number
            b = record[1]  # second number

        and if any of the variables are arrays, the shape can be specified as
        the third item in the relevant dtype::

            double precision :: a
            integer :: b(3,4)
            write(1) a, b

            record = f.read_record('<f4', np.dtype(('<i4', (4, 3))))
            a = record[0]
            b = record[1].T

        NumPy also supports a short syntax for this kind of type::

            record = f.read_record('<f4', '(3,3)<i4')

        See Also
        --------
        read_reals
        read_ints

        """
    dtype = kwargs.pop('dtype', None)
    if kwargs:
        raise ValueError(f'Unknown keyword arguments {tuple(kwargs.keys())}')
    if dtype is not None:
        dtypes = dtypes + (dtype,)
    elif not dtypes:
        raise ValueError('Must specify at least one dtype')
    first_size = self._read_size(eof_ok=True)
    dtypes = tuple((np.dtype(dtype) for dtype in dtypes))
    block_size = sum((dtype.itemsize for dtype in dtypes))
    num_blocks, remainder = divmod(first_size, block_size)
    if remainder != 0:
        raise ValueError(f'Size obtained ({first_size}) is not a multiple of the dtypes given ({block_size}).')
    if len(dtypes) != 1 and first_size != block_size:
        raise ValueError(f'Size obtained ({first_size}) does not match with the expected size ({block_size}) of multi-item record')
    data = []
    for dtype in dtypes:
        r = np.fromfile(self._fp, dtype=dtype, count=num_blocks)
        if len(r) != num_blocks:
            raise FortranFormattingError('End of file in the middle of a record')
        if dtype.shape != ():
            if num_blocks == 1:
                assert r.shape == (1,) + dtype.shape
                r = r[0]
        data.append(r)
    second_size = self._read_size()
    if first_size != second_size:
        raise ValueError('Sizes do not agree in the header and footer for this record - check header dtype')
    if len(dtypes) == 1:
        return data[0]
    else:
        return tuple(data)