from __future__ import absolute_import, division, print_function
import os
import mimetypes
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def container_has_blobs(self):
    try:
        blobs = self.blob_service_client.get_container_client(container=self.container).list_blobs()
    except Exception as exc:
        self.fail('Error list blobs in {0} - {1}'.format(self.container, str(exc)))
    if len(list(blobs)) > 0:
        return True
    return False