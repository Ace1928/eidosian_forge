from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule, json
from ansible_collections.cisco.meraki.plugins.module_utils.network.meraki.meraki import MerakiModule, meraki_argument_spec
def get_admin(meraki, data, id):
    for a in data:
        if a['id'] == id:
            return a
    meraki.fail_json(msg='No admin found by specified name or email')