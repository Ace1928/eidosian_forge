from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
import uuid
def get_cdn_client(self):
    if not self.cdn_client:
        self.cdn_client = self.get_mgmt_svc_client(CdnManagementClient, base_url=self._cloud_environment.endpoints.resource_manager, is_track2=True, api_version='2017-04-02')
    return self.cdn_client