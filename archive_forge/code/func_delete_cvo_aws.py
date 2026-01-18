from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp import CloudManagerRestAPI
def delete_cvo_aws(self, we_id):
    """
        Delete AWS CVO
        """
    api_url = '%s/working-environments/%s' % (self.rest_api.api_root_path, we_id)
    response, error, on_cloud_request_id = self.rest_api.delete(api_url, None, header=self.headers)
    if error is not None:
        self.module.fail_json(msg='Error: unexpected response on deleting cvo aws: %s, %s' % (str(error), str(response)))
    wait_on_completion_api_url = '/occm/api/audit/activeTask/%s' % str(on_cloud_request_id)
    err = self.rest_api.wait_on_completion(wait_on_completion_api_url, 'CVO', 'delete', 40, 60)
    if err is not None:
        self.module.fail_json(msg='Error: unexpected response wait_on_completion for deleting CVO AWS: %s' % str(err))