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
def destroy_vm(self, vm_name):
    vm = self._lookup_vm_check(vm_name, as_vssd=False)
    job_path, ret_val = self._vs_man_svc.DestroySystem(vm.path_())
    self._jobutils.check_ret_val(ret_val, job_path)