from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
import uuid
def cdnprofile_to_dict(cdnprofile):
    return dict(id=cdnprofile.id, name=cdnprofile.name, type=cdnprofile.type, location=cdnprofile.location, sku=cdnprofile.sku.name, resource_state=cdnprofile.resource_state, provisioning_state=cdnprofile.provisioning_state, tags=cdnprofile.tags)