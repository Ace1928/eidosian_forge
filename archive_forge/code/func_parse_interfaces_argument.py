from __future__ import absolute_import, division, print_function
import re
from time import sleep
import itertools
from copy import deepcopy
from time import sleep
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import NetworkConfig
from ansible_collections.community.network.plugins.module_utils.network.icx.icx import load_config, get_config
from ansible.module_utils.connection import Connection, ConnectionError, exec_command
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import conditional, remove_default_spec
def parse_interfaces_argument(module, item, port_type):
    untagged_ports, untagged_lags, tagged_ports, tagged_lags = parse_vlan_brief(module, item)
    ports = list()
    if port_type == 'interfaces':
        if untagged_ports:
            for port in untagged_ports:
                ports.append('ethernet 1/1/' + str(port))
        if untagged_lags:
            for port in untagged_lags:
                ports.append('lag ' + str(port))
    elif port_type == 'tagged':
        if tagged_ports:
            for port in tagged_ports:
                ports.append('ethernet 1/1/' + str(port))
        if tagged_lags:
            for port in tagged_lags:
                ports.append('lag ' + str(port))
    return ports