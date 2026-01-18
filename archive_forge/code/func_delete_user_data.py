from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.scaleway import SCALEWAY_LOCATION, scaleway_argument_spec, Scaleway
def delete_user_data(compute_api, server_id, key):
    compute_api.module.debug('Starting deleting user_data attributes: %s' % key)
    response = compute_api.delete(path='servers/%s/user_data/%s' % (server_id, key))
    if not response.ok:
        msg = ('Error during user_data deleting: (%s) %s' % response.status_code, response.body)
        compute_api.module.fail_json(msg=msg)
    return response