from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def build_create_payload(self, want, commands):
    temp = {}
    if 'session_timeout' in commands and commands['session_timeout'] is not None:
        temp['session-timeout'] = commands['session_timeout']
    if 'keepalive' in commands and commands['keepalive'] is not None:
        temp['keepalive-interval'] = commands['keepalive']
    if 'source_address' in commands and commands['source_address'] is not None:
        temp['source-address'] = commands['source_address']
    if 'peer_address' in commands and commands['peer_address'] is not None:
        temp['peer-address'] = commands['peer_address']
    if 'peer_link' in commands and commands['peer_link'] is not None:
        temp['peer-link'] = str(commands['peer_link'])
    if 'system_mac' in commands and commands['system_mac'] is not None:
        temp['openconfig-mclag:mclag-system-mac'] = str(commands['system_mac'])
    if 'delay_restore' in commands and commands['delay_restore'] is not None:
        temp['delay-restore'] = commands['delay_restore']
    mclag_dict = {}
    if temp:
        domain_id = {'domain-id': want['domain_id']}
        mclag_dict.update(domain_id)
        config = {'config': temp}
        mclag_dict.update(config)
        payload = {'openconfig-mclag:mclag-domain': [mclag_dict]}
    else:
        payload = {}
    return payload