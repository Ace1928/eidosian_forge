from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.scaleway import SCALEWAY_LOCATION, scaleway_argument_spec, Scaleway
from ansible.module_utils.basic import AnsibleModule
def get_nics_info(api, compute_id, private_network_id):
    response = api.get('servers/' + compute_id + '/private_nics')
    if not response.ok:
        msg = "Error during get servers information: %s: '%s' (%s)" % (response.info['msg'], response.json['message'], response.json)
        api.module.fail_json(msg=msg)
    i = 0
    list_nics = response.json['private_nics']
    while i < len(list_nics):
        if list_nics[i]['private_network_id'] == private_network_id:
            return list_nics[i]
        i += 1
    return None