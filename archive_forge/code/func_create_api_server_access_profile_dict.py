from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def create_api_server_access_profile_dict(api_server):
    return api_server.as_dict() if api_server else dict()