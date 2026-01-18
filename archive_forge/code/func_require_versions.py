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
def require_versions(requires):
    """ Utility function for consolidating multiple `gi.require_version()` calls.

    :param requires: The names and versions of modules to require.
    :type requires: dict

    :Example:

    .. code-block:: python

        import gi
        gi.require_versions({'Gtk': '3.0', 'GLib': '2.0', 'Gio': '2.0'})
    """
    for module_name, module_version in requires.items():
        require_version(module_name, module_version)