from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, json
from ansible_collections.cisco.meraki.plugins.module_utils.network.meraki.meraki import MerakiModule, meraki_argument_spec
from copy import deepcopy
def does_stack_exist(meraki, stacks):
    for stack in stacks:
        have = set(meraki.params['serials'])
        want = set(stack['serials'])
        if have == want:
            return stack
    return False