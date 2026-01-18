import os
import time
import socket
import pathlib
import tempfile
import contextlib
from typing import Union, Optional
from functools import lru_cache
@lru_cache()
def is_readonly_dir(path: Union[str, pathlib.Path]) -> bool:
    """
    Check if a directory is read-only.
    """
    if isinstance(path, str):
        path = pathlib.Path(path)
    if not path.is_dir():
        path = path.parent
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile('w', dir=path.as_posix()) as f:
            f.write('test')
        return False
    except Exception as e:
        return True