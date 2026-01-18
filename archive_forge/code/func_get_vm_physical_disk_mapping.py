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
def get_vm_physical_disk_mapping(self, vm_name, is_planned_vm=False):
    mapping = {}
    physical_disks = self.get_vm_disks(vm_name)[1]
    for diskdrive in physical_disks:
        mapping[diskdrive.ElementName] = dict(resource_path=diskdrive.path_(), mounted_disk_path=diskdrive.HostResource[0])
    return mapping