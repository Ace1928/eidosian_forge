from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, normalize_location_name
def resource_group_to_dict(rg):
    return dict(id=rg.id, name=rg.name, location=rg.location, tags=rg.tags, provisioning_state=rg.properties.provisioning_state)