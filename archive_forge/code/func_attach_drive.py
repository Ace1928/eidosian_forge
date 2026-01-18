import functools
import time
import uuid
from eventlet import patcher
from eventlet import tpool
from oslo_log import log as logging
from oslo_utils import uuidutils
from six.moves import range  # noqa
from os_win._i18n import _
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils import baseutils
from os_win.utils import jobutils
from os_win.utils import pathutils
def attach_drive(self, vm_name, path, ctrller_path, drive_addr, drive_type=constants.DISK):
    """Create a drive and attach it to the vm."""
    vm = self._lookup_vm_check(vm_name, as_vssd=False)
    if drive_type == constants.DISK:
        res_sub_type = self._DISK_DRIVE_RES_SUB_TYPE
    elif drive_type == constants.DVD:
        res_sub_type = self._DVD_DRIVE_RES_SUB_TYPE
    drive = self._get_new_resource_setting_data(res_sub_type)
    drive.Parent = ctrller_path
    drive.Address = drive_addr
    drive.AddressOnParent = drive_addr
    new_resources = self._jobutils.add_virt_resource(drive, vm)
    drive_path = new_resources[0]
    if drive_type == constants.DISK:
        res_sub_type = self._HARD_DISK_RES_SUB_TYPE
    elif drive_type == constants.DVD:
        res_sub_type = self._DVD_DISK_RES_SUB_TYPE
    res = self._get_new_resource_setting_data(res_sub_type, self._STORAGE_ALLOC_SETTING_DATA_CLASS)
    res.Parent = drive_path
    res.HostResource = [path]
    try:
        self._jobutils.add_virt_resource(res, vm)
    except Exception:
        LOG.exception('Failed to attach disk image %(disk_path)s to vm %(vm_name)s. Reverting attachment.', dict(disk_path=path, vm_name=vm_name))
        drive = self._get_wmi_obj(drive_path)
        self._jobutils.remove_virt_resource(drive)
        raise