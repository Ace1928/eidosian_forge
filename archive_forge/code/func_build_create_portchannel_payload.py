from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def build_create_portchannel_payload(self, want, commands):
    payload = {'openconfig-mclag:interface': []}
    for each in commands:
        payload['openconfig-mclag:interface'].append({'name': each['lag'], 'config': {'name': each['lag'], 'mclag-domain-id': want['domain_id']}})
    return payload