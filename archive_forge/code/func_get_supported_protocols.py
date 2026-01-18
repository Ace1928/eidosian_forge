from __future__ import absolute_import, division, print_function
import json
import re
from ansible.errors import AnsibleConnectionFailure
from ansible.module_utils._text import to_text
from ansible.module_utils.common._collections_compat import Mapping
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list
from ansible_collections.ansible.netcommon.plugins.plugin_utils.cliconf_base import (
def get_supported_protocols(self):
    supported_protocols = {}
    protocols = ['bgp', 'isis', 'ospf', 'ldp', 'ospf6', 'pim', 'rip', 'ripm', 'zebra']
    daemons = self.get('show daemons')
    data = to_text(daemons, errors='surrogate_or_strict').strip()
    for item in protocols:
        supported_protocols[item] = True if item in data else False
    return supported_protocols