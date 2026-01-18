import logging
import weakref
from _dbus_bindings import (
from dbus.connection import Connection
from dbus.exceptions import DBusException
from dbus.lowlevel import HANDLER_RESULT_NOT_YET_HANDLED
from dbus._compat import is_py2
def error_cb(e):
    if e.get_dbus_name() == _NAME_HAS_NO_OWNER:
        callback('')
    else:
        logging.basicConfig()
        _logger.debug('GetNameOwner(%s) failed:', bus_name, exc_info=(e.__class__, e, None))