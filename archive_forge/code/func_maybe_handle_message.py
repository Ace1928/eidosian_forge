import logging
import threading
import weakref
from _dbus_bindings import (
from dbus.exceptions import DBusException
from dbus.lowlevel import (
from dbus.proxies import ProxyObject
from dbus._compat import is_py2, is_py3
from _dbus_bindings import String
def maybe_handle_message(self, message):
    args = None
    if self._sender_name_owner not in (None, message.get_sender()):
        return False
    if self._int_args_match is not None:
        kwargs = dict(byte_arrays=True)
        args = message.get_args_list(**kwargs)
        for index, value in self._int_args_match.items():
            if index >= len(args) or not isinstance(args[index], String) or args[index] != value:
                return False
    if self._member not in (None, message.get_member()):
        return False
    if self._interface not in (None, message.get_interface()):
        return False
    if self._path not in (None, message.get_path()):
        return False
    try:
        if args is None or not self._byte_arrays:
            args = message.get_args_list(byte_arrays=self._byte_arrays)
        kwargs = {}
        if self._sender_keyword is not None:
            kwargs[self._sender_keyword] = message.get_sender()
        if self._destination_keyword is not None:
            kwargs[self._destination_keyword] = message.get_destination()
        if self._path_keyword is not None:
            kwargs[self._path_keyword] = message.get_path()
        if self._member_keyword is not None:
            kwargs[self._member_keyword] = message.get_member()
        if self._interface_keyword is not None:
            kwargs[self._interface_keyword] = message.get_interface()
        if self._message_keyword is not None:
            kwargs[self._message_keyword] = message
        self._handler(*args, **kwargs)
    except:
        logging.basicConfig()
        _logger.error('Exception in handler for D-Bus signal:', exc_info=1)
    return True