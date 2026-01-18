import warnings
import numpy
import cupy
def savez_compressed(file, *args, **kwds):
    """Saves one or more arrays into a file in compressed ``.npz`` format.

    It is equivalent to :func:`cupy.savez` function except the output file is
    compressed.

    .. seealso::
       :func:`cupy.savez` for more detail,
       :func:`numpy.savez_compressed`

    """
    args = map(cupy.asnumpy, args)
    for key in kwds:
        kwds[key] = cupy.asnumpy(kwds[key])
    numpy.savez_compressed(file, *args, **kwds)