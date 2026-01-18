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
def attach_ide_drive(self, vm_name, path, ctrller_addr, drive_addr, drive_type=constants.DISK):
    vmsettings = self._lookup_vm_check(vm_name)
    ctrller_path = self._get_vm_ide_controller(vmsettings, ctrller_addr)
    self.attach_drive(vm_name, path, ctrller_path, drive_addr, drive_type)