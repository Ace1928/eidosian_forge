from __future__ import annotations
import os
import pathlib
from typing import Sequence
def filefind(filename: str, path_dirs: Sequence[str] | None=None) -> str:
    """Find a file by looking through a sequence of paths.

    This iterates through a sequence of paths looking for a file and returns
    the full, absolute path of the first occurrence of the file.  If no set of
    path dirs is given, the filename is tested as is, after running through
    :func:`expandvars` and :func:`expanduser`.  Thus a simple call::

        filefind('myfile.txt')

    will find the file in the current working dir, but::

        filefind('~/myfile.txt')

    Will find the file in the users home directory.  This function does not
    automatically try any paths, such as the cwd or the user's home directory.

    Parameters
    ----------
    filename : str
        The filename to look for.
    path_dirs : str, None or sequence of str
        The sequence of paths to look for the file in.  If None, the filename
        need to be absolute or be in the cwd.  If a string, the string is
        put into a sequence and the searched.  If a sequence, walk through
        each element and join with ``filename``, calling :func:`expandvars`
        and :func:`expanduser` before testing for existence.

    Returns
    -------
    Raises :exc:`IOError` or returns absolute path to file.
    """
    filename = filename.strip('"').strip("'")
    if os.path.isabs(filename) and os.path.isfile(filename):
        return filename
    if path_dirs is None:
        path_dirs = ('',)
    elif isinstance(path_dirs, str):
        path_dirs = (path_dirs,)
    elif isinstance(path_dirs, pathlib.Path):
        path_dirs = (str(path_dirs),)
    for path in path_dirs:
        if path == '.':
            path = os.getcwd()
        testname = expand_path(os.path.join(path, filename))
        if os.path.isfile(testname):
            return os.path.abspath(testname)
    raise OSError(f'File {filename!r} does not exist in any of the search paths: {path_dirs!r}')