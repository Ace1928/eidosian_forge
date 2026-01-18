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
@_utils.retry_decorator(exceptions=exceptions.WMIJobFailed)
def set_vm_state(self, vm_name, req_state):
    """Set the desired state of the VM."""
    vm = self._lookup_vm_check(vm_name, as_vssd=False)
    job_path, ret_val = vm.RequestStateChange(self._vm_power_states_map[req_state])
    self._jobutils.check_ret_val(ret_val, job_path, [0, 32775])
    LOG.debug('Successfully changed vm state of %(vm_name)s to %(req_state)s', {'vm_name': vm_name, 'req_state': req_state})