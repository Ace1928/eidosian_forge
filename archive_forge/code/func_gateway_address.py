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
def gateway_address(self):
    if self._values['gateway_address'] is None:
        return None
    try:
        if '%' in self._values['gateway_address']:
            addr = self._values['gateway_address'].split('%')[0]
            ip_interface(u'%s' % str(addr))
        else:
            addr = self._values['gateway_address']
            ip_interface(u'%s' % str(addr))
            if self.route_domain:
                result = str(addr).lower() + '%' + str(self.route_domain)
                return result
        return str(self._values['gateway_address']).lower()
    except ValueError:
        raise F5ModuleError('The provided gateway_address is not an IP address')