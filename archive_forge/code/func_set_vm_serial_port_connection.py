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
def set_vm_serial_port_connection(self, vm_name, port_number, pipe_path):
    vmsettings = self._lookup_vm_check(vm_name)
    serial_port = self._get_vm_serial_ports(vmsettings)[port_number - 1]
    serial_port.Connection = [pipe_path]
    self._jobutils.modify_virt_resource(serial_port)