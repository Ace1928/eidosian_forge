from __future__ import absolute_import, division, print_function
import copy
import os
import ssl
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.connection import exec_command
from ..module_utils.common import (
def restart_daemon(self):
    cmd = 'tmsh restart /sys service httpd'
    rc, out, err = exec_command(self.module, cmd)
    if rc != 0:
        raise F5ModuleError(err)