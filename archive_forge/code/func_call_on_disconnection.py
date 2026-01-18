import logging
import threading
import weakref
from _dbus_bindings import (
from dbus.exceptions import DBusException
from dbus.lowlevel import (
from dbus.proxies import ProxyObject
from dbus._compat import is_py2, is_py3
from _dbus_bindings import String
def call_on_disconnection(self, callable):
    """Arrange for `callable` to be called with one argument (this
        Connection object) when the Connection becomes
        disconnected.

        :Since: 0.83.0
        """
    self.__call_on_disconnection.append(callable)