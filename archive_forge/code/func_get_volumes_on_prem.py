from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp import CloudManagerRestAPI
def get_volumes_on_prem(self, working_environment_detail, name):
    response, err, dummy = self.rest_api.send_request('GET', '/occm/api/onprem/volumes?workingEnvironmentId=%s&name=%s' % (working_environment_detail['publicId'], name), None, header=self.headers)
    if err is not None:
        self.module.fail_json(changed=False, msg='Error getting volume on prem %s: %s.' % (err, response))
    return response