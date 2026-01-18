from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def curated_items(self, raws):
    return [self.zone_to_dict(item) for item in raws] if raws else []