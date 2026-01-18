from __future__ import absolute_import, division, print_function
import ansible_collections.netapp.storagegrid.plugins.module_utils.netapp as netapp_utils
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp import SGRestAPI
def get_org_user_s3_key(self, user_id, access_key):
    api = 'api/v3/org/users/current-user/s3-access-keys/%s' % access_key
    if user_id:
        api = 'api/v3/org/users/%s/s3-access-keys/%s' % (user_id, access_key)
    response, error = self.rest_api.get(api)
    if error:
        self.module.fail_json(msg=error)
    else:
        return response['data']
    return None