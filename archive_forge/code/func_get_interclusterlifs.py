from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp import CloudManagerRestAPI
def get_interclusterlifs(self, source_we_id, dest_we_id):
    api_get = '/occm/api/replication/intercluster-lifs?peerWorkingEnvironmentId=%s&workingEnvironmentId=%s' % (dest_we_id, source_we_id)
    response, err, dummy_second = self.rest_api.send_request('GET', api_get, None, header=self.headers)
    if err is not None:
        self.module.fail_json(changed=False, msg='Error getting interclusterlifs %s: %s.' % (err, response))
    return response