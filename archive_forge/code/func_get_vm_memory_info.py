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
def get_vm_memory_info(self, vm_name):
    vmsetting = self._lookup_vm_check(vm_name)
    memory = self._get_vm_memory(vmsetting)
    memory_info_dict = {'DynamicMemoryEnabled': memory.DynamicMemoryEnabled, 'Reservation': memory.Reservation, 'Limit': memory.Limit, 'Weight': memory.Weight, 'MaxMemoryBlocksPerNumaNode': memory.MaxMemoryBlocksPerNumaNode}
    return memory_info_dict