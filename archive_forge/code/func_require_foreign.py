from pkgutil import extend_path
import sys
import os
import importlib
import types
from . import _gi
from ._gi import _API  # noqa: F401
from ._gi import Repository
from ._gi import PyGIDeprecationWarning  # noqa: F401
from ._gi import PyGIWarning  # noqa: F401
def require_foreign(namespace, symbol=None):
    """Ensure the given foreign marshaling module is available and loaded.

    :param str namespace:
        Introspection namespace of the foreign module (e.g. "cairo")
    :param symbol:
        Optional symbol typename to ensure a converter exists.
    :type symbol: str or None
    :raises: ImportError

    :Example:

    .. code-block:: python

        import gi
        import cairo
        gi.require_foreign('cairo')

    """
    try:
        _gi.require_foreign(namespace, symbol)
    except Exception as e:
        raise ImportError(str(e))
    importlib.import_module('gi.repository', namespace)