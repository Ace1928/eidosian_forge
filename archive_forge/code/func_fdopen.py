import abc
import sys
import stat as st
from _collections_abc import _check_methods
from os.path import (curdir, pardir, sep, pathsep, defpath, extsep, altsep,
from _collections_abc import MutableMapping, Mapping
def fdopen(fd, mode='r', buffering=-1, encoding=None, *args, **kwargs):
    if not isinstance(fd, int):
        raise TypeError('invalid fd type (%s, expected integer)' % type(fd))
    import io
    if 'b' not in mode:
        encoding = io.text_encoding(encoding)
    return io.open(fd, mode, buffering, encoding, *args, **kwargs)