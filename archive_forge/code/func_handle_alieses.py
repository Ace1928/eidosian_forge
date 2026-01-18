from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.facts.facts import Facts
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.rm_templates.snmp_server import (
def handle_alieses(self, want):
    for x in [want.get('groups', []), want.get('users', [])]:
        for y in x:
            if y.get('Ipv4_acl'):
                del y['Ipv4_acl']
            if y.get('Ipv6_acl'):
                del y['Ipv6_acl']
    return want