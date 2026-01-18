from __future__ import absolute_import, division, print_function
import copy
import os
import ssl
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.connection import exec_command
from ..module_utils.common import (
def read_current_certificate(self):
    result = dict()
    command = 'openssl x509 -in /config/httpd/conf/ssl.crt/{0} -dates -issuer -noout'.format(self.want.cert_name)
    rc, out, err = exec_command(self.module, command)
    if rc != 0:
        raise F5ModuleError(err)
    if rc == 0:
        result['epoch'] = self._parse_cert_date(out)
    return ApiParameters(params=result)