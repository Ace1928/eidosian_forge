from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def is_azure_default_restriction(self, restriction_obj):
    return restriction_obj['action'] == 'Allow' and restriction_obj['ip_address'] == 'Any' and (restriction_obj['priority'] == 1) or (restriction_obj['action'] == 'Deny' and restriction_obj['ip_address'] == 'Any' and (restriction_obj['priority'] == 2147483647))