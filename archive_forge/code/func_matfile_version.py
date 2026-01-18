import numpy as np
from scipy._lib import doccer
from . import _byteordercodes as boc
def matfile_version(file_name, *, appendmat=True):
    """
    Return major, minor tuple depending on apparent mat file type

    Where:

     #. 0,x -> version 4 format mat files
     #. 1,x -> version 5 format mat files
     #. 2,x -> version 7.3 format mat files (HDF format)

    Parameters
    ----------
    file_name : str
       Name of the mat file (do not need .mat extension if
       appendmat==True). Can also pass open file-like object.
    appendmat : bool, optional
       True to append the .mat extension to the end of the given
       filename, if not already present. Default is True.

    Returns
    -------
    major_version : {0, 1, 2}
        major MATLAB File format version
    minor_version : int
        minor MATLAB file format version

    Raises
    ------
    MatReadError
        If the file is empty.
    ValueError
        The matfile version is unknown.

    Notes
    -----
    Has the side effect of setting the file read pointer to 0
    """
    from ._mio import _open_file_context
    with _open_file_context(file_name, appendmat=appendmat) as fileobj:
        return _get_matfile_version(fileobj)