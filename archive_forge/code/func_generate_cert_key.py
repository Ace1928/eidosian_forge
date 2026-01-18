from __future__ import absolute_import, division, print_function
import copy
import os
import ssl
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.connection import exec_command
from ..module_utils.common import (
def generate_cert_key(self):
    cmd = 'openssl req -x509 -nodes -days {3} -newkey rsa:{4} -keyout {0}/ssl.key/{2} -out {0}/ssl.crt/{1} -subj "{5}"'.format('/config/httpd/conf', self.want.cert_name, self.want.key_name, self.want.days_valid, self.want.key_size, self.want.issuer)
    rc, out, err = exec_command(self.module, cmd)
    if rc != 0:
        raise F5ModuleError(err)