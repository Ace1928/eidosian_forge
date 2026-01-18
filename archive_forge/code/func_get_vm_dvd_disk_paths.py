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
def get_vm_dvd_disk_paths(self, vm_name):
    vmsettings = self._lookup_vm_check(vm_name)
    sasds = _wqlutils.get_element_associated_class(self._conn, self._STORAGE_ALLOC_SETTING_DATA_CLASS, element_instance_id=vmsettings.InstanceID)
    dvd_paths = [sasd.HostResource[0] for sasd in sasds if sasd.ResourceSubType == self._DVD_DISK_RES_SUB_TYPE]
    return dvd_paths