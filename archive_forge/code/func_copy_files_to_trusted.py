from __future__ import absolute_import, division, print_function
import copy
import os
import ssl
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.connection import exec_command
from ..module_utils.common import (
def copy_files_to_trusted(self):
    cmd1 = 'cat /config/httpd/conf/ssl.crt/{0} >> /config/big3d/client.crt'.format(self.want.cert_name)
    cmd2 = 'cat /config/httpd/conf/ssl.crt/{0} >> /config/gtm/server.crt'.format(self.want.cert_name)
    rc, out, err = exec_command(self.module, cmd1)
    if rc != 0:
        raise F5ModuleError(err)
    rc, out, err = exec_command(self.module, cmd2)
    if rc != 0:
        raise F5ModuleError(err)