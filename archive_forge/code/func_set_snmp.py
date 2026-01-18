from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, json
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.cisco.meraki.plugins.module_utils.network.meraki.meraki import MerakiModule, meraki_argument_spec
def set_snmp(meraki, org_id):
    payload = dict()
    if meraki.params['v2c_enabled'] is not None:
        payload = {'v2cEnabled': meraki.params['v2c_enabled']}
    if meraki.params['v3_enabled'] is True:
        if len(meraki.params['v3_auth_pass']) < 8 or len(meraki.params['v3_priv_pass']) < 8:
            meraki.fail_json(msg='v3_auth_pass and v3_priv_pass must both be at least 8 characters long.')
        if meraki.params['v3_auth_mode'] is None or meraki.params['v3_auth_pass'] is None or meraki.params['v3_priv_mode'] is None or (meraki.params['v3_priv_pass'] is None):
            meraki.fail_json(msg='v3_auth_mode, v3_auth_pass, v3_priv_mode, and v3_auth_pass are required')
        payload = {'v3Enabled': meraki.params['v3_enabled'], 'v3AuthMode': meraki.params['v3_auth_mode'].upper(), 'v3AuthPass': meraki.params['v3_auth_pass'], 'v3PrivMode': meraki.params['v3_priv_mode'].upper(), 'v3PrivPass': meraki.params['v3_priv_pass']}
        if meraki.params['peer_ips'] is not None:
            payload['peerIps'] = meraki.params['peer_ips']
    elif meraki.params['v3_enabled'] is False:
        payload = {'v3Enabled': False}
    full_compare = snake_dict_to_camel_dict(payload)
    path = meraki.construct_path('create', org_id=org_id)
    snmp = get_snmp(meraki, org_id)
    ignored_parameters = ['v3AuthPass', 'v3PrivPass', 'hostname', 'port', 'v2CommunityString', 'v3User']
    if meraki.is_update_required(snmp, full_compare, optional_ignore=ignored_parameters):
        if meraki.module.check_mode is True:
            meraki.generate_diff(snmp, full_compare)
            snmp.update(payload)
            meraki.result['data'] = snmp
            meraki.result['changed'] = True
            meraki.exit_json(**meraki.result)
        r = meraki.request(path, method='PUT', payload=json.dumps(payload))
        if meraki.status == 200:
            meraki.generate_diff(snmp, r)
            meraki.result['changed'] = True
            return r
    else:
        return snmp