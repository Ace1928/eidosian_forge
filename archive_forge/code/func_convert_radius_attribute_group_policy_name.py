from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, json
from ansible_collections.cisco.meraki.plugins.module_utils.network.meraki.meraki import (
def convert_radius_attribute_group_policy_name(arg):
    if arg == 'Filter-Id':
        return 11
    else:
        return ''