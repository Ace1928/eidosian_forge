from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, json
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.cisco.meraki.plugins.module_utils.network.meraki.meraki import MerakiModule, meraki_argument_spec
def get_snmp(meraki, org_id):
    path = meraki.construct_path('get_all', org_id=org_id)
    r = meraki.request(path, method='GET')
    if meraki.status == 200:
        return r