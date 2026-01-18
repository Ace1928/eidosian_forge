import time
from oslo_log import log as logging
from os_win import _utils
import os_win.conf
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
@_utils.not_found_decorator()
@_utils.retry_decorator(exceptions=exceptions.HyperVException)
def add_virt_resource(self, virt_resource, parent):
    job_path, new_resources, ret_val = self._vs_man_svc.AddResourceSettings(parent.path_(), [virt_resource.GetText_(1)])
    self.check_ret_val(ret_val, job_path)
    return new_resources