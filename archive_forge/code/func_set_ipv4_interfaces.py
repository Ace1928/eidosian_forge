from __future__ import absolute_import, division, print_function
import re
from ansible_collections.community.network.plugins.module_utils.network.enos.enos import run_commands, enos_argument_spec, check_args
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible.module_utils.six.moves import zip
def set_ipv4_interfaces(self, line4):
    ipv4_addresses = list()
    for line in line4:
        ipv4Split = line.split()
        if ipv4Split[1] == 'IP4':
            ipv4_addresses.append(ipv4Split[2])
    return ipv4_addresses