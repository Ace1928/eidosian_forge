import sys
from os import fspath, fsencode, fsdecode
from ..version import hdf5_built_version_tuple
def filename_encode(filename):
    """
    Encode filename for use in the HDF5 library.

    Due to how HDF5 handles filenames on different systems, this should be
    called on any filenames passed to the HDF5 library. See the documentation on
    filenames in h5py for more information.
    """
    filename = fspath(filename)
    if sys.platform == 'win32':
        if isinstance(filename, str):
            return filename.encode(WINDOWS_ENCODING, 'strict')
        return filename
    return fsencode(filename)