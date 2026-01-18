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
def dst_datacenter(self):
    dst_datacenter = self._values['destination'].get('datacenter', None)
    if dst_datacenter is None:
        return None
    return fq_name(self.partition, dst_datacenter)