from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule, json
from ansible_collections.cisco.meraki.plugins.module_utils.network.meraki.meraki import MerakiModule, meraki_argument_spec
def create_admin(meraki, org_id, name, email):
    payload = dict()
    payload['name'] = name
    payload['email'] = email
    is_admin_existing = find_admin(meraki, get_admins(meraki, org_id), email)
    if meraki.params['org_access'] is not None:
        payload['orgAccess'] = meraki.params['org_access']
    if meraki.params['tags'] is not None:
        payload['tags'] = meraki.params['tags']
    if meraki.params['networks'] is not None:
        nets = meraki.get_nets(org_id=org_id)
        networks = network_factory(meraki, meraki.params['networks'], nets)
        payload['networks'] = networks
    if is_admin_existing is None:
        if meraki.module.check_mode is True:
            meraki.result['data'] = payload
            meraki.result['changed'] = True
            meraki.exit_json(**meraki.result)
        path = meraki.construct_path('create', function='admin', org_id=org_id)
        r = meraki.request(path, method='POST', payload=json.dumps(payload))
        if meraki.status == 201:
            meraki.result['changed'] = True
            return r
    elif is_admin_existing is not None:
        if not meraki.params['tags']:
            payload['tags'] = []
        if not meraki.params['networks']:
            payload['networks'] = []
        if meraki.is_update_required(is_admin_existing, payload) is True:
            if meraki.module.check_mode is True:
                meraki.generate_diff(is_admin_existing, payload)
                is_admin_existing.update(payload)
                meraki.result['changed'] = True
                meraki.result['data'] = payload
                meraki.exit_json(**meraki.result)
            path = meraki.construct_path('update', function='admin', org_id=org_id) + is_admin_existing['id']
            r = meraki.request(path, method='PUT', payload=json.dumps(payload))
            if meraki.status == 200:
                meraki.result['changed'] = True
                return r
        else:
            meraki.result['data'] = is_admin_existing
            if meraki.module.check_mode is True:
                meraki.result['data'] = payload
                meraki.exit_json(**meraki.result)
            return -1