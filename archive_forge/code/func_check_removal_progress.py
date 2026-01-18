from __future__ import absolute_import, division, print_function
import time
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def check_removal_progress(self):
    done = self.check_progress()
    if done:
        if self.should_reboot():
            self.save_on_device()
            self.changes.update({'message': 'Device finished de-provisioning requested module: {0} and configuration has been saved, a reboot is required.'.format(self.want.module)})
        return True
    return False