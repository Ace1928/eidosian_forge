from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def has_access_restrictions(self, site_config):
    return site_config.ip_security_restrictions or site_config.scm_ip_security_restrictions