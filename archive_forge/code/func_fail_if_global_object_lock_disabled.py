from __future__ import absolute_import, division, print_function
import ansible_collections.netapp.storagegrid.plugins.module_utils.netapp as netapp_utils
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp import SGRestAPI
def fail_if_global_object_lock_disabled(self):
    api = 'api/v3/org/compliance-global'
    response, error = self.rest_api.get(api)
    if error:
        self.module.fail_json(msg=error)
    if not response['data']['complianceEnabled']:
        self.module.fail_json(msg='Error: Global S3 Object Lock setting is not enabled.')