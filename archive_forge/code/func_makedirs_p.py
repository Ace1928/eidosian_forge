from __future__ import annotations
import errno
import os
from contextlib import contextmanager
from typing import TYPE_CHECKING
def makedirs_p(path: Union[str, Path], **kwargs) -> None:
    """
    Wrapper for os.makedirs that does not raise an exception if the directory
    already exists, in the fashion of "mkdir -p" command. The check is
    performed in a thread-safe way

    Args:
        path: path of the directory to create
        kwargs: standard kwargs for os.makedirs
    """
    try:
        os.makedirs(path, **kwargs)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise