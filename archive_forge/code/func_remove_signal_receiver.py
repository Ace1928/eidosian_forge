import logging
import threading
import weakref
from _dbus_bindings import (
from dbus.exceptions import DBusException
from dbus.lowlevel import (
from dbus.proxies import ProxyObject
from dbus._compat import is_py2, is_py3
from _dbus_bindings import String
def remove_signal_receiver(self, handler_or_match, signal_name=None, dbus_interface=None, bus_name=None, path=None, **keywords):
    named_service = keywords.pop('named_service', None)
    if named_service is not None:
        if bus_name is not None:
            raise TypeError('bus_name and named_service cannot both be specified')
        bus_name = named_service
        from warnings import warn
        warn('Passing the named_service parameter to remove_signal_receiver by name is deprecated: please use positional parameters', DeprecationWarning, stacklevel=2)
    new = []
    deletions = []
    self._signals_lock.acquire()
    try:
        by_interface = self._signal_recipients_by_object_path.get(path, None)
        if by_interface is None:
            return
        by_member = by_interface.get(dbus_interface, None)
        if by_member is None:
            return
        matches = by_member.get(signal_name, None)
        if matches is None:
            return
        for match in matches:
            if handler_or_match is match or match.matches_removal_spec(bus_name, path, dbus_interface, signal_name, handler_or_match, **keywords):
                deletions.append(match)
            else:
                new.append(match)
        if new:
            by_member[signal_name] = new
        else:
            del by_member[signal_name]
            if not by_member:
                del by_interface[dbus_interface]
                if not by_interface:
                    del self._signal_recipients_by_object_path[path]
    finally:
        self._signals_lock.release()
    for match in deletions:
        self._clean_up_signal_match(match)