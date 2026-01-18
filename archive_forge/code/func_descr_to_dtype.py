import numpy
import warnings
from numpy.lib.utils import safe_eval, drop_metadata
from numpy.compat import (
def descr_to_dtype(descr):
    """
    Returns a dtype based off the given description.

    This is essentially the reverse of `dtype_to_descr()`. It will remove
    the valueless padding fields created by, i.e. simple fields like
    dtype('float32'), and then convert the description to its corresponding
    dtype.

    Parameters
    ----------
    descr : object
        The object retrieved by dtype.descr. Can be passed to
        `numpy.dtype()` in order to replicate the input dtype.

    Returns
    -------
    dtype : dtype
        The dtype constructed by the description.

    """
    if isinstance(descr, str):
        return numpy.dtype(descr)
    elif isinstance(descr, tuple):
        dt = descr_to_dtype(descr[0])
        return numpy.dtype((dt, descr[1]))
    titles = []
    names = []
    formats = []
    offsets = []
    offset = 0
    for field in descr:
        if len(field) == 2:
            name, descr_str = field
            dt = descr_to_dtype(descr_str)
        else:
            name, descr_str, shape = field
            dt = numpy.dtype((descr_to_dtype(descr_str), shape))
        is_pad = name == '' and dt.type is numpy.void and (dt.names is None)
        if not is_pad:
            title, name = name if isinstance(name, tuple) else (None, name)
            titles.append(title)
            names.append(name)
            formats.append(dt)
            offsets.append(offset)
        offset += dt.itemsize
    return numpy.dtype({'names': names, 'formats': formats, 'titles': titles, 'offsets': offsets, 'itemsize': offset})