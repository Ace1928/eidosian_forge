from __future__ import absolute_import, division, print_function
import os
import time
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def policy_import(self):
    self._set_changed_options()
    if self.module.check_mode:
        return True
    if self.exists():
        if self.want.force is False:
            return False
    if not self.exists() and self.want.force is True:
        self.want.update({'force': None})
    if self.want.inline:
        task = self.inline_import()
        self.wait_for_task(task)
        return True
    self._clear_changes()
    self.import_file_to_device()
    return True