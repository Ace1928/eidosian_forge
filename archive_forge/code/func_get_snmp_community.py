from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def get_snmp_community(module, name):
    command = 'show run snmp all | grep word-exp {0}'.format(name)
    data = execute_show_command(command, module)[0]
    community_dict = {}
    if not data:
        return community_dict
    community_re = 'snmp-server community (\\S+)'
    mo = re.search(community_re, data)
    if mo:
        community_name = mo.group(1)
    else:
        return community_dict
    community_dict['group'] = None
    group_re = 'snmp-server community {0} group (\\S+)'.format(community_name)
    mo = re.search(group_re, data)
    if mo:
        community_dict['group'] = mo.group(1)
    community_dict['acl'] = None
    acl_re = 'snmp-server community {0} use-acl (\\S+)'.format(community_name)
    mo = re.search(acl_re, data)
    if mo:
        community_dict['acl'] = mo.group(1)
    return community_dict