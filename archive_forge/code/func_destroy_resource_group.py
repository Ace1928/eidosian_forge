from __future__ import absolute_import, division, print_function
import time
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def destroy_resource_group(self):
    """
        Destroy the targeted resource group
        """
    try:
        result = self.rm_client.resource_groups.begin_delete(self.resource_group)
        result.wait()
    except Exception as e:
        if e.status_code == 404 or e.status_code == 204:
            return
        else:
            self.fail('Delete resource group and deploy failed with status code: %s and message: %s' % (e.status_code, e.message))