from __future__ import annotations
from typing import TYPE_CHECKING
def get_ipython() -> 'InteractiveShell':
    """
    Return running IPython instance or None
    """
    try:
        from IPython.core.getipython import get_ipython as _get_ipython
    except ImportError as err:
        raise type(err)('IPython is has not been installed.') from err
    ip = _get_ipython()
    if ip is None:
        raise RuntimeError('Not running in a juptyer session.')
    return ip