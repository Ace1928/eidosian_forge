from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def curated_list(self, raws):
    return [self.record_to_dict(item) for item in raws] if raws else []