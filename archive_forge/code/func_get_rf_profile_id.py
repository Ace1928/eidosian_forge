from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, json
from ansible_collections.cisco.meraki.plugins.module_utils.network.meraki.meraki import (
from re import sub
def get_rf_profile_id(meraki):
    """Get the RF Profile ID for a given RF Profile Name."""
    profile_id = meraki.params['rf_profile_id']
    profile_name = meraki.params['rf_profile_name']
    if profile_id is None and profile_name is not None:
        net_id = get_net_id(meraki)
        path = meraki.construct_path('get_all', 'mr_rf_profile', net_id=net_id)
        profiles = meraki.request(path, method='GET')
        profile_id = next((profile['id'] for profile in profiles if profile['name'] == meraki.params['rf_profile_name']), None)
    return profile_id