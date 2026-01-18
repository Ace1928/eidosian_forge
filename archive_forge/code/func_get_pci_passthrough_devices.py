import socket
from oslo_log import log as logging
from os_win._i18n import _
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
from os_win.utils.winapi import libs as w_lib
def get_pci_passthrough_devices(self):
    """Get host PCI devices path.

        Discrete device assignment is supported only on Windows / Hyper-V
        Server 2016 or newer.

        :returns: a list of the assignable PCI devices.
        """
    return []