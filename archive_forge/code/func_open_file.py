from .. import utils
from .._lazyload import h5py
from .._lazyload import tables
from decorator import decorator
@with_HDF5
def open_file(filename, mode='r', backend=None):
    """Open an HDF5 file with either tables or h5py.

    Gives a simple, unified interface for both tables and h5py

    Parameters
    ----------
    filename : str
        Name of the HDF5 file
    mode : str, optional (default: 'r')
        Read/write mode. Choose from ['r', 'w', 'a' 'r+']
    backend : str, optional (default: None)
        HDF5 backend to use. Choose from ['h5py', 'tables']. If not given,
        scprep will detect which backend is available, using tables if
        both are installed.

    Returns
    -------
    f : tables.File or h5py.File
        Open HDF5 file handle.
    """
    if backend is None:
        if utils._try_import('tables'):
            backend = 'tables'
        else:
            backend = 'h5py'
    if backend == 'tables':
        return tables.open_file(filename, mode)
    elif backend == 'h5py':
        return h5py.File(filename, mode)
    else:
        raise ValueError("Expected backend in ['tables', 'h5py']. Got {}".format(backend))