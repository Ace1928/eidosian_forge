import os
import sys
import py
import tempfile
def dupfile(f, mode=None, buffering=0, raising=False, encoding=None):
    """ return a new open file object that's a duplicate of f

        mode is duplicated if not given, 'buffering' controls
        buffer size (defaulting to no buffering) and 'raising'
        defines whether an exception is raised when an incompatible
        file object is passed in (if raising is False, the file
        object itself will be returned)
    """
    try:
        fd = f.fileno()
        mode = mode or f.mode
    except AttributeError:
        if raising:
            raise
        return f
    newfd = os.dup(fd)
    if sys.version_info >= (3, 0):
        if encoding is not None:
            mode = mode.replace('b', '')
            buffering = True
        return os.fdopen(newfd, mode, buffering, encoding, closefd=True)
    else:
        f = os.fdopen(newfd, mode, buffering)
        if encoding is not None:
            return EncodedFile(f, encoding)
        return f