from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.ipaddress import is_valid_ip_network
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def dst_state(self):
    dst_country = self.dst_country
    dst_state = self._values['destination'].get('state', None)
    if dst_state is None:
        return None
    if dst_country is None:
        raise F5ModuleError('Country needs to be provided when specifying state')
    result = '{0}/{1}'.format(dst_country, dst_state)
    return result