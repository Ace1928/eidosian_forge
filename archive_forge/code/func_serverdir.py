from __future__ import annotations
import logging # isort:skip
from pathlib import Path
from .deprecation import deprecated
def serverdir() -> str:
    """ Get the location of the server subpackage.

    .. deprecated:: 3.4.0
        Use ``server_path()`` instead.
    """
    deprecated((3, 4, 0), 'serverdir()', 'server_path()')
    return str(server_path())