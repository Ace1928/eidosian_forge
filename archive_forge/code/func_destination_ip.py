from __future__ import absolute_import, division, print_function
import re
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.parsing.convert_bool import BOOLEANS_TRUE
from ipaddress import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.ipaddress import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def destination_ip(self):
    if self._values['destination'] is None:
        return None
    destination = self.destination_to_network()
    try:
        pattern = '(?P<rd>%[0-9]+)'
        addr = re.sub(pattern, '', destination)
        ip = ip_network(u'%s' % str(addr))
        return '{0}/{1}'.format(str(ip.network_address), ip.prefixlen)
    except ValueError:
        raise F5ModuleError('The provided destination is not an IP address.')