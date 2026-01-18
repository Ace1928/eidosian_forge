from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp import CloudManagerRestAPI
def get_aggregates_info(self, rest_api, headers):
    """
        Get aggregates info: there are 4 types of working environments.
        Each of the aggregates will be categorized by working environment type and working environment id
        """
    aggregates = {}
    working_environments, error = self.na_helper.get_working_environments_info(rest_api, headers)
    if error is not None:
        self.module.fail_json(msg='Error: Failed to get working environments: %s' % str(error))
    for working_env_type in working_environments:
        we_aggregates = {}
        for we in working_environments[working_env_type]:
            provider = we['cloudProviderName']
            working_environment_id = we['publicId']
            self.na_helper.set_api_root_path(we, rest_api)
            if provider != 'Amazon':
                api = '%s/aggregates/%s' % (rest_api.api_root_path, working_environment_id)
            else:
                api = '%s/aggregates?workingEnvironmentId=%s' % (rest_api.api_root_path, working_environment_id)
            response, error, dummy = rest_api.get(api, None, header=headers)
            if error:
                self.module.fail_json(msg='Error: Failed to get aggregate list: %s' % str(error))
            we_aggregates[working_environment_id] = response
        aggregates[working_env_type] = we_aggregates
    return aggregates