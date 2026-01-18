from __future__ import annotations
import io
import sys
import typing as ty
import warnings
from functools import reduce
from operator import getitem, mul
from os.path import exists, splitext
import numpy as np
from ._compression import COMPRESSED_FILE_LIKES
from .casting import OK_FLOATS, shared_range
from .externals.oset import OrderedSet
def array_from_file(shape: tuple[int, ...], in_dtype: np.dtype[DT], infile: io.IOBase, offset: int=0, order: ty.Literal['C', 'F']='F', mmap: bool | ty.Literal['c', 'r', 'r+']=True) -> npt.NDArray[DT]:
    """Get array from file with specified shape, dtype and file offset

    Parameters
    ----------
    shape : sequence
        sequence specifying output array shape
    in_dtype : numpy dtype
        fully specified numpy dtype, including correct endianness
    infile : file-like
        open file-like object implementing at least read() and seek()
    offset : int, optional
        offset in bytes into `infile` to start reading array data. Default is 0
    order : {'F', 'C'} string
        order in which to write data.  Default is 'F' (fortran order).
    mmap : {True, False, 'c', 'r', 'r+'}
        `mmap` controls the use of numpy memory mapping for reading data.  If
        False, do not try numpy ``memmap`` for data array.  If one of {'c',
        'r', 'r+'}, try numpy memmap with ``mode=mmap``.  A `mmap` value of
        True gives the same behavior as ``mmap='c'``.  If `infile` cannot be
        memory-mapped, ignore `mmap` value and read array from file.

    Returns
    -------
    arr : array-like
        array like object that can be sliced, containing data

    Examples
    --------
    >>> from io import BytesIO
    >>> bio = BytesIO()
    >>> arr = np.arange(6).reshape(1,2,3)
    >>> _ = bio.write(arr.tobytes('F'))  # outputs int
    >>> arr2 = array_from_file((1,2,3), arr.dtype, bio)
    >>> np.all(arr == arr2)
    True
    >>> bio = BytesIO()
    >>> _ = bio.write(b' ' * 10)
    >>> _ = bio.write(arr.tobytes('F'))
    >>> arr2 = array_from_file((1,2,3), arr.dtype, bio, 10)
    >>> np.all(arr == arr2)
    True
    """
    if mmap not in (True, False, 'c', 'r', 'r+'):
        raise ValueError("mmap value should be one of True, False, 'c', 'r', 'r+'")
    in_dtype = np.dtype(in_dtype)
    infile = getattr(infile, 'fobj', infile)
    if mmap and (not _is_compressed_fobj(infile)):
        mode = 'c' if mmap is True else mmap
        try:
            return np.memmap(infile, in_dtype, mode=mode, shape=shape, order=order, offset=offset)
        except (AttributeError, TypeError, ValueError):
            pass
    if len(shape) == 0:
        return np.array([], in_dtype)
    n_bytes = reduce(mul, shape) * in_dtype.itemsize
    if n_bytes == 0:
        return np.array([], in_dtype)
    infile.seek(offset)
    if hasattr(infile, 'readinto'):
        data_bytes = bytearray(n_bytes)
        n_read = infile.readinto(data_bytes)
        needs_copy = False
    else:
        data_bytes = infile.read(n_bytes)
        n_read = len(data_bytes)
        needs_copy = True
    if n_bytes != n_read:
        raise OSError(f'Expected {n_bytes} bytes, got {n_read} bytes from {getattr(infile, 'name', 'object')}\n - could the file be damaged?')
    arr: np.ndarray = np.ndarray(shape, in_dtype, buffer=data_bytes, order=order)
    if needs_copy:
        return arr.copy()
    arr.flags.writeable = True
    return arr