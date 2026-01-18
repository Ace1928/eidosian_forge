from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.basic import AnsibleModule, json
from ansible_collections.cisco.meraki.plugins.module_utils.network.meraki.meraki import (
def sort_roles(syslog_servers):
    """Accept a full payload and sort roles"""
    for i, server in enumerate(syslog_servers):
        syslog_servers['servers'][i]['roles'] = sorted(syslog_servers['servers'][i]['roles'])