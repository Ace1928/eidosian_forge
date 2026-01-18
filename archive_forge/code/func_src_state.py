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
def src_state(self):
    src_country = self._values['source'].get('country', None)
    src_state = self._values['source'].get('state', None)
    if src_state is None:
        return None
    if src_country is None:
        raise F5ModuleError('Country needs to be provided when specifying state')
    result = '{0}/{1}'.format(src_country, src_state)
    return result