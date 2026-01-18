from __future__ import absolute_import, division, print_function
import copy
import os
import ssl
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.connection import exec_command
from ..module_utils.common import (
def configure_new_cert(self):
    cmd1 = 'tmsh modify sys httpd ssl-certkeyfile /config/httpd/conf/ssl.key/{1} ssl-certfile /config/httpd/conf/ssl.crt/{0}'.format(self.want.cert_name, self.want.key_name)
    cmd2 = 'tmsh save /sys config partitions all'
    rc, out, err = exec_command(self.module, cmd1)
    if rc != 0:
        raise F5ModuleError(err)
    rc, out, err = exec_command(self.module, cmd2)
    if rc != 0:
        raise F5ModuleError(err)