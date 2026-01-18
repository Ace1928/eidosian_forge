import logging
import threading
import weakref
from _dbus_bindings import (
from dbus.exceptions import DBusException
from dbus.lowlevel import (
from dbus.proxies import ProxyObject
from dbus._compat import is_py2, is_py3
from _dbus_bindings import String
def matches_removal_spec(self, sender, object_path, dbus_interface, member, handler, **kwargs):
    if handler not in (None, self._handler):
        return False
    if sender != self._sender:
        return False
    if object_path != self._path:
        return False
    if dbus_interface != self._interface:
        return False
    if member != self._member:
        return False
    if kwargs != self._args_match:
        return False
    return True