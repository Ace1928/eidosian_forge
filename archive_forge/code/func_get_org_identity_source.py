from __future__ import absolute_import, division, print_function
import ansible_collections.netapp.storagegrid.plugins.module_utils.netapp as netapp_utils
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp import SGRestAPI
def get_org_identity_source(self):
    api = 'api/v3/org/identity-source'
    response, error = self.rest_api.get(api)
    if error:
        self.module.fail_json(msg=error)
    else:
        return response['data']
    return None