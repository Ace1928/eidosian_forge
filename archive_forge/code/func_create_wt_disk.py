from oslo_log import log as logging
import six
from os_win._i18n import _
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
from os_win.utils import hostutils
from os_win.utils import pathutils
from os_win.utils import win32utils
def create_wt_disk(self, vhd_path, wtd_name, size_mb=None):
    try:
        self._conn_wmi.WT_Disk.NewWTDisk(DevicePath=vhd_path, Description=wtd_name, SizeInMB=size_mb)
    except exceptions.x_wmi as wmi_exc:
        err_msg = _('Failed to create WT Disk. VHD path: %(vhd_path)s WT disk name: %(wtd_name)s')
        raise exceptions.ISCSITargetWMIException(err_msg % dict(vhd_path=vhd_path, wtd_name=wtd_name), wmi_exc=wmi_exc)