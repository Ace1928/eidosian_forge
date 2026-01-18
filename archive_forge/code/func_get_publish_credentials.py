from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def get_publish_credentials(self, resource_group, name):
    self.log('Get web app {0} publish credentials'.format(name))
    try:
        poller = self.web_client.web_apps.begin_list_publishing_credentials(resource_group_name=resource_group, name=name)
        if isinstance(poller, LROPoller):
            response = self.get_poller_result(poller)
    except Exception as ex:
        request_id = ex.request_id if ex.request_id else ''
        self.fail('Error getting web app {0} publishing credentials - {1}'.format(request_id, str(ex)))
    return response