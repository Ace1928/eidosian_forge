from __future__ import absolute_import, division, print_function
import os
import re
from copy import deepcopy
from datetime import datetime
from ansible.module_utils.urls import urlparse
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import remove_default_spec
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.ipaddress import is_valid_ip, validate_ip_v6_address
from ..module_utils.compare import cmp_str_with_none
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def compare_addresses(self, items):
    if any(('address' in item for item in items)):
        aggregates = [self._join_address_port(item) for item in items if 'address' in item and item['address']]
        collection = [member['name'] for member in self.on_device]
        diff = set(collection) - set(aggregates)
        if diff:
            addresses = [item['selfLink'] for item in self.on_device if item['name'] in diff]
            self.purge_links.extend(addresses)
            return True
        return False
    return False