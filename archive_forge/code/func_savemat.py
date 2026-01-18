from contextlib import contextmanager
from ._miobase import _get_matfile_version, docfiller
from ._mio4 import MatFile4Reader, MatFile4Writer
from ._mio5 import MatFile5Reader, MatFile5Writer
@docfiller
def savemat(file_name, mdict, appendmat=True, format='5', long_field_names=False, do_compression=False, oned_as='row'):
    """
    Save a dictionary of names and arrays into a MATLAB-style .mat file.

    This saves the array objects in the given dictionary to a MATLAB-
    style .mat file.

    Parameters
    ----------
    file_name : str or file-like object
        Name of the .mat file (.mat extension not needed if ``appendmat ==
        True``).
        Can also pass open file_like object.
    mdict : dict
        Dictionary from which to save matfile variables.
    appendmat : bool, optional
        True (the default) to append the .mat extension to the end of the
        given filename, if not already present.
    format : {'5', '4'}, string, optional
        '5' (the default) for MATLAB 5 and up (to 7.2),
        '4' for MATLAB 4 .mat files.
    long_field_names : bool, optional
        False (the default) - maximum field name length in a structure is
        31 characters which is the documented maximum length.
        True - maximum field name length in a structure is 63 characters
        which works for MATLAB 7.6+.
    do_compression : bool, optional
        Whether or not to compress matrices on write. Default is False.
    oned_as : {'row', 'column'}, optional
        If 'column', write 1-D NumPy arrays as column vectors.
        If 'row', write 1-D NumPy arrays as row vectors.

    Examples
    --------
    >>> from scipy.io import savemat
    >>> import numpy as np
    >>> a = np.arange(20)
    >>> mdic = {"a": a, "label": "experiment"}
    >>> mdic
    {'a': array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19]),
    'label': 'experiment'}
    >>> savemat("matlab_matrix.mat", mdic)
    """
    with _open_file_context(file_name, appendmat, 'wb') as file_stream:
        if format == '4':
            if long_field_names:
                message = 'Long field names are not available for version 4 files'
                raise ValueError(message)
            MW = MatFile4Writer(file_stream, oned_as)
        elif format == '5':
            MW = MatFile5Writer(file_stream, do_compression=do_compression, unicode_strings=True, long_field_names=long_field_names, oned_as=oned_as)
        else:
            raise ValueError("Format should be '4' or '5'")
        MW.put_variables(mdict)