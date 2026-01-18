from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp import CloudManagerRestAPI
def get_aggregate_detail(self, working_environment_detail, aggregate_name):
    if working_environment_detail['workingEnvironmentType'] == 'ON_PREM':
        api = '/occm/api/onprem/aggregates?workingEnvironmentId=%s' % working_environment_detail['publicId']
    else:
        self.na_helper.set_api_root_path(working_environment_detail, self.rest_api)
        api_root_path = self.rest_api.api_root_path
        if working_environment_detail['cloudProviderName'] != 'Amazon':
            api = '%s/aggregates/%s'
        else:
            api = '%s/aggregates?workingEnvironmentId=%s'
        api = api % (api_root_path, working_environment_detail['publicId'])
    response, error, dummy = self.rest_api.get(api, header=self.headers)
    if error:
        self.module.fail_json(msg='Error: Failed to get aggregate list: %s' % str(error))
    for aggr in response:
        if aggr['name'] == aggregate_name:
            return aggr
    return None