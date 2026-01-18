from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def create_or_update_source_control(self):
    """
        Update site source control
        :return: deserialized updating response
        """
    self.log('Update site source control')
    if self.deployment_source is None:
        return False
    self.deployment_source['is_manual_integration'] = False
    self.deployment_source['is_mercurial'] = False
    try:
        site_source_control = SiteSourceControl(repo_url=self.deployment_source.get('url'), branch=self.deployment_source.get('branch'))
        response = self.web_client.web_apps.begin_create_or_update_source_control(resource_group_name=self.resource_group, name=self.name, site_source_control=site_source_control)
        self.log('Response : {0}'.format(response))
        return response.as_dict()
    except Exception:
        self.fail('Failed to update site source control for web app {0} in resource group {1}'.format(self.name, self.resource_group))