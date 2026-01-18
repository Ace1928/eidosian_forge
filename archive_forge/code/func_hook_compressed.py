import io
import sys, os
from types import GenericAlias
def hook_compressed(filename, mode, *, encoding=None, errors=None):
    if encoding is None and 'b' not in mode:
        encoding = 'locale'
    ext = os.path.splitext(filename)[1]
    if ext == '.gz':
        import gzip
        stream = gzip.open(filename, mode)
    elif ext == '.bz2':
        import bz2
        stream = bz2.BZ2File(filename, mode)
    else:
        return open(filename, mode, encoding=encoding, errors=errors)
    if 'b' not in mode:
        stream = io.TextIOWrapper(stream, encoding=encoding, errors=errors)
    return stream