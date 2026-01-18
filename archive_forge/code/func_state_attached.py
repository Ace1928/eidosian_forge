from __future__ import absolute_import, division, print_function
import os
import platform
import re
import tempfile
import time
from ansible.module_utils.basic import AnsibleModule
def state_attached(self):
    if not self.exists():
        self.msg.append('zone does not exist')
    if self.is_configured():
        self.attach()
    else:
        self.msg.append('zone already attached')