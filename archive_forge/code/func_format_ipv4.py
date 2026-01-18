from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.argspec.l3_interfaces.l3_interfaces import (
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.utils.utils import (
def format_ipv4(self, address):
    if address.split(' ')[1]:
        cidr_val = netmask_to_cidr(address.split(' ')[1])
    return address.split(' ')[0] + '/' + cidr_val