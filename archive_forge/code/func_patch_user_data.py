from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.scaleway import SCALEWAY_LOCATION, scaleway_argument_spec, Scaleway
def patch_user_data(compute_api, server_id, key, value):
    compute_api.module.debug('Starting patching user_data attributes')
    path = 'servers/%s/user_data/%s' % (server_id, key)
    response = compute_api.patch(path=path, data=value, headers={'Content-Type': 'text/plain'})
    if not response.ok:
        msg = 'Error during user_data patching: %s %s' % (response.status_code, response.body)
        compute_api.module.fail_json(msg=msg)
    return response