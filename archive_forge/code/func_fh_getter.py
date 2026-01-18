import collections
from pathlib import Path
import string
from urllib.request import urlopen
import warnings
from cartopy import config
def fh_getter(fh, mode='r', needs_filename=False):
    """
    Convenience function for opening files.

    Parameters
    ----------
    fh
        File handle, filename or (file handle, filename) tuple.
    mode: optional
        Open mode. Defaults to "r".
    needs_filename: optional
        Defaults to False

    Returns
    -------
    file handle, filename
        Opened in the given mode.

    """
    if mode != 'r':
        raise ValueError('Only mode "r" currently supported.')
    if isinstance(fh, str):
        filename = fh
        fh = open(fh, mode)
    elif isinstance(fh, tuple):
        fh, filename = fh
    if filename is None:
        try:
            filename = fh.name
        except AttributeError:
            if needs_filename:
                raise ValueError('filename cannot be determined')
            else:
                filename = ''
    return (fh, filename)