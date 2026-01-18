from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import (
from ..module_utils.icontrol import (
from ..module_utils.ipaddress import ip_network
from ..module_utils.teem import send_teem
@property
def dtls_port(self):
    if self._values['dtls_port'] is None:
        return None
    if 0 < self._values['dtls_port'] > 65535:
        raise F5ModuleError('Specified port number is out of valid range, correct range is between 0 and 65535.')
    return self._values['dtls_port']