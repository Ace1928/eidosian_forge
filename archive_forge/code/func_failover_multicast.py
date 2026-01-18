from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.ipaddress import is_valid_ip
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def failover_multicast(self):
    values = ['multicast_address', 'multicast_interface', 'multicast_port']
    if self.want.failover_multicast is False:
        if self.have.multicast_interface == 'eth0' and self.have.multicast_address == 'any' and (self.have.multicast_port == 0):
            return None
        else:
            result = dict(failover_multicast=True, multicast_port=0, multicast_interface='eth0', multicast_address='any')
            return result
    elif all((self.have._values[x] in [None, 'any6', 'any'] for x in values)):
        return True