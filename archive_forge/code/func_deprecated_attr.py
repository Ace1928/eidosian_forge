import functools
import types
import warnings
import importlib
import sys
from gi import PyGIDeprecationWarning
from gi._gi import CallableInfo, pygobject_new_full
from gi._constants import \
from pkgutil import extend_path
def deprecated_attr(namespace, attr, replacement):
    """Marks a module level attribute as deprecated. Accessing it will emit
    a PyGIDeprecationWarning warning.

    e.g. for ``deprecated_attr("GObject", "STATUS_FOO", "GLib.Status.FOO")``
    accessing GObject.STATUS_FOO will emit:

        "GObject.STATUS_FOO is deprecated; use GLib.Status.FOO instead"

    :param str namespace:
        The namespace of the override this is called in.
    :param str namespace:
        The attribute name (which gets added to __all__).
    :param str replacement:
        The replacement text which will be included in the warning.
    """
    _deprecated_attrs.setdefault(namespace, []).append((attr, replacement))