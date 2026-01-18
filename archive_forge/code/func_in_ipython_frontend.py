from __future__ import annotations
from shutil import get_terminal_size
def in_ipython_frontend() -> bool:
    """
    Check if we're inside an IPython zmq frontend.

    Returns
    -------
    bool
    """
    try:
        ip = get_ipython()
        return 'zmq' in str(type(ip)).lower()
    except NameError:
        pass
    return False