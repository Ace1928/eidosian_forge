from __future__ import absolute_import, division, print_function
import os
import platform
import re
import tempfile
import time
from ansible.module_utils.basic import AnsibleModule
def state_stopped(self):
    if self.exists():
        self.stop()
    else:
        self.module.fail_json(msg='zone does not exist')