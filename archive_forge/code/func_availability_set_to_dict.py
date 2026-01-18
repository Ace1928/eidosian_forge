from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def availability_set_to_dict(avaset):
    """
    Serializing the availability set from the API to Dict
    :return: dict
    """
    return dict(id=avaset.id, name=avaset.name, location=avaset.location, platform_update_domain_count=avaset.platform_update_domain_count, platform_fault_domain_count=avaset.platform_fault_domain_count, proximity_placement_group=avaset.proximity_placement_group.id if avaset.proximity_placement_group else None, tags=avaset.tags, sku=avaset.sku.name)