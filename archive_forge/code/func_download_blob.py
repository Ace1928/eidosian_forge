from __future__ import absolute_import, division, print_function
import os
import mimetypes
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def download_blob(self):
    if not self.check_mode:
        try:
            client = self.blob_service_client.get_blob_client(container=self.container, blob=self.blob)
            with open(self.dest, 'wb') as blob_stream:
                blob_data = client.download_blob()
                blob_data.readinto(blob_stream)
        except Exception as exc:
            self.fail('Failed to download blob {0}:{1} to {2} - {3}'.format(self.container, self.blob, self.dest, exc))
    self.results['changed'] = True
    self.results['actions'].append('downloaded blob {0}:{1} to {2}'.format(self.container, self.blob, self.dest))
    self.results['container'] = self.container_obj
    self.results['blob'] = self.blob_obj