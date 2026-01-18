import numpy
import warnings
from numpy.lib.utils import safe_eval, drop_metadata
from numpy.compat import (
def read_magic(fp):
    """ Read the magic string to get the version of the file format.

    Parameters
    ----------
    fp : filelike object

    Returns
    -------
    major : int
    minor : int
    """
    magic_str = _read_bytes(fp, MAGIC_LEN, 'magic string')
    if magic_str[:-2] != MAGIC_PREFIX:
        msg = 'the magic string is not correct; expected %r, got %r'
        raise ValueError(msg % (MAGIC_PREFIX, magic_str[:-2]))
    major, minor = magic_str[-2:]
    return (major, minor)