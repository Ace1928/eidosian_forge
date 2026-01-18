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
def dst_country(self):
    dst_country = self._values['destination'].get('country', None)
    if dst_country is None:
        return None
    result = self.countries.get(dst_country, dst_country)
    return result