from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBase
def application_to_dict(self, object):
    return dict(app_id=object.app_id, object_id=object.object_id, display_name=object.display_name)