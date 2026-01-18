import warnings
import numpy
import cupy
def savez(file, *args, **kwds):
    """Saves one or more arrays into a file in uncompressed ``.npz`` format.

    Arguments without keys are treated as arguments with automatic keys named
    ``arr_0``, ``arr_1``, etc. corresponding to the positions in the argument
    list. The keys of arguments are used as keys in the ``.npz`` file, which
    are used for accessing NpzFile object when the file is read by
    :func:`cupy.load` function.

    Args:
        file (file or str): File or filename to save.
        *args: Arrays with implicit keys.
        **kwds: Arrays with explicit keys.

    .. seealso:: :func:`numpy.savez`

    """
    args = map(cupy.asnumpy, args)
    for key in kwds:
        kwds[key] = cupy.asnumpy(kwds[key])
    numpy.savez(file, *args, **kwds)