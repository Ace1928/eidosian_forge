from __future__ import absolute_import, division, print_function
import os
import platform
import re
import tempfile
import time
from ansible.module_utils.basic import AnsibleModule
def state_configured(self):
    if self.exists():
        self.msg.append('zone already exists')
    else:
        self.configure()