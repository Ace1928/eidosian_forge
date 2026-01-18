from __future__ import absolute_import, division, print_function
import os
import mimetypes
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def delete_blob(self):
    if not self.check_mode:
        try:
            self.blob_service_client.get_container_client(container=self.container).delete_blob(blob=self.blob)
        except Exception as exc:
            self.fail('Error deleting blob {0}:{1} - {2}'.format(self.container, self.blob, str(exc)))
    self.results['changed'] = True
    self.results['actions'].append('deleted blob {0}:{1}'.format(self.container, self.blob))
    self.results['container'] = self.container_obj