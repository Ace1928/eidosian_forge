import socket
from oslo_log import log as logging
from os_win._i18n import _
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
from os_win.utils.winapi import libs as w_lib
def get_volume_info(self, drive):
    """Returns a tuple with total size and free space of the given drive.

        Returned values are expressed in bytes.

        :param drive: the drive letter of the logical disk whose information
            is required.
        """
    logical_disk = self._conn_cimv2.query("SELECT Size, FreeSpace FROM win32_logicaldisk WHERE DeviceID='%s'" % drive)[0]
    return (int(logical_disk.Size), int(logical_disk.FreeSpace))