from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule, json
from ansible_collections.cisco.meraki.plugins.module_utils.network.meraki.meraki import MerakiModule, meraki_argument_spec
def get_admins(meraki, org_id):
    admins = meraki.request(meraki.construct_path('query', function='admin', org_id=org_id), method='GET')
    if meraki.status == 200:
        return admins