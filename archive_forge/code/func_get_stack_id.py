from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, json
from ansible_collections.cisco.meraki.plugins.module_utils.network.meraki.meraki import MerakiModule, meraki_argument_spec
from copy import deepcopy
def get_stack_id(meraki, net_id):
    stacks = get_stacks(meraki, net_id)
    for stack in stacks:
        if stack['name'] == meraki.params['name']:
            return stack['id']