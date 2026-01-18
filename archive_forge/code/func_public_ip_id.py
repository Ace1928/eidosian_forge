from __future__ import absolute_import, division, print_function
import time
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from copy import deepcopy
from ansible.module_utils.common.dict_transformations import (
def public_ip_id(subscription_id, resource_group_name, name):
    """Generate the id for a frontend ip configuration"""
    return '/subscriptions/{0}/resourceGroups/{1}/providers/Microsoft.Network/publicIPAddresses/{2}'.format(subscription_id, resource_group_name, name)