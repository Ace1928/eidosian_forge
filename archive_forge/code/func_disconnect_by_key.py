from __future__ import annotations
import itertools
import typing
import warnings
import weakref
def disconnect_by_key(self, obj, name: Hashable, key: Key) -> None:
    """
        :param obj: the object to disconnect the signal from
        :type obj: object
        :param name: the signal to disconnect, typically a string
        :type name: signal name
        :param key: the key for this signal handler, as returned by
                    connect_signal().
        :type key: Key

        This function will remove a callback from the list connected
        to a signal with connect_signal(). The key passed should be the
        value returned by connect_signal().

        If the callback is not connected or already disconnected, this
        function will simply do nothing.
        """
    handlers = setdefaultattr(obj, self._signal_attr, {}).get(name, [])
    handlers[:] = [h for h in handlers if h[0] is not key]