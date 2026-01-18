import time
from oslo_log import log as logging
from os_win import _utils
import os_win.conf
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
@_utils.not_found_decorator()
def remove_multiple_virt_features(self, virt_features):
    job_path, ret_val = self._vs_man_svc.RemoveFeatureSettings(FeatureSettings=[f.path_() for f in virt_features])
    self.check_ret_val(ret_val, job_path)