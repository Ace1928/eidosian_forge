from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _snake_to_camel
def default_content_types():
    return ['text/plain', 'text/html', 'text/css', 'text/javascript', 'application/x-javascript', 'application/javascript', 'application/json', 'application/xml']