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
def get_vm_storage_paths(self, vm_name, is_planned_vm=False):
    vmsettings = self._lookup_vm_check(vm_name)
    disk_resources, volume_resources = self._get_vm_disks(vmsettings)
    volume_drives = []
    for volume_resource in volume_resources:
        drive_path = volume_resource.HostResource[0]
        volume_drives.append(drive_path)
    disk_files = []
    for disk_resource in disk_resources:
        disk_files.extend([c for c in self._get_disk_resource_disk_path(disk_resource)])
    return (disk_files, volume_drives)