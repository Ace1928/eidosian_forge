from __future__ import absolute_import, division, print_function
import copy
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AZURE_SUCCESS_STATE, AzureRMModuleBase
from ansible.module_utils._text import to_native
def set_blob_cors(self):
    try:
        cors_rules = self.storage_models.CorsRules(cors_rules=[self.storage_models.CorsRule(**x) for x in self.blob_cors])
        self.storage_client.blob_services.set_service_properties(self.resource_group, self.name, self.storage_models.BlobServiceProperties(cors=cors_rules))
    except Exception as exc:
        self.fail('Failed to set CORS rules: {0}'.format(str(exc)))