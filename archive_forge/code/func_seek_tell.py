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
def seek_tell(fileobj: io.IOBase, offset: int, write0: bool=False) -> None:
    """Seek in `fileobj` or check we're in the right place already

    Parameters
    ----------
    fileobj : file-like
        object implementing ``seek`` and (if seek raises an OSError) ``tell``
    offset : int
        position in file to which to seek
    write0 : {False, True}, optional
        If True, and standard seek fails, try to write zeros to the file to
        reach `offset`.  This can be useful when writing bz2 files, that cannot
        do write seeks.
    """
    try:
        fileobj.seek(offset)
    except OSError as e:
        pos = fileobj.tell()
        if pos == offset:
            return
        if not write0:
            raise OSError(str(e))
        if pos > offset:
            raise OSError("Can't write to seek backwards")
        fileobj.write(b'\x00' * (offset - pos))
        assert fileobj.tell() == offset