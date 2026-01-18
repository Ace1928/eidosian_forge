from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, azure_id_to_dict
def serialize_nics(self, raws):
    return [self.serialize_obj(item, AZURE_OBJECT_CLASS) for item in raws] if raws else []