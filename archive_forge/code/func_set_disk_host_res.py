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
def set_disk_host_res(self, disk_res_path, mounted_disk_path):
    diskdrive = self._get_wmi_obj(disk_res_path, True)
    diskdrive.HostResource = [mounted_disk_path]
    self._jobutils.modify_virt_resource(diskdrive)