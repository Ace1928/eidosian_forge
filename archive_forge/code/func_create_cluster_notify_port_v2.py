import contextlib
import ctypes
from os_win._i18n import _
from os_win import constants
from os_win import exceptions
from os_win.utils import win32utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import libs as w_lib
from os_win.utils.winapi.libs import clusapi as clusapi_def
from os_win.utils.winapi import wintypes
def create_cluster_notify_port_v2(self, cluster_handle, notif_filters, notif_port_h=None, notif_key=None):
    """Creates or updates a cluster notify port.

        This allows us to subscribe to specific types of cluster events.

        :param cluster_handle: an open cluster handle, for which we'll
                               receive events. This handle must remain open
                               while fetching events.
        :param notif_filters: an array of NOTIFY_FILTER_AND_TYPE structures,
                              specifying the event types we're listening to.
        :param notif_port_h: an open cluster notify port handle, when adding
                             new filters to an existing cluster notify port,
                             or INVALID_HANDLE_VALUE when creating a new
                             notify port.
        :param notif_key: a DWORD value that will be mapped to a specific
                          event type. When fetching events, the cluster API
                          will send us back a reference to the according
                          notification key. For this reason, we must ensure
                          that this variable will not be garbage collected
                          while waiting for events.
        :return: the requested notify port handle,
        """
    notif_port_h = notif_port_h or w_const.INVALID_HANDLE_VALUE
    notif_filters_len = len(notif_filters) if isinstance(notif_filters, ctypes.Array) else 1
    notif_key_p = ctypes.byref(notif_key) if notif_key is not None else None
    notif_port_h = self._run_and_check_output(clusapi.CreateClusterNotifyPortV2, notif_port_h, cluster_handle, ctypes.byref(notif_filters), ctypes.c_ulong(notif_filters_len), notif_key_p, **self._open_handle_check_flags)
    return notif_port_h