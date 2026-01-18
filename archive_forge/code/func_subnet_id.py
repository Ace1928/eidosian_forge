from __future__ import absolute_import, division, print_function
import time
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from copy import deepcopy
from ansible.module_utils.common.dict_transformations import (
def subnet_id(subscription_id, resource_group_name, virtual_network_name, name):
    """Generate the id for a subnet in a virtual network"""
    return '/subscriptions/{0}/resourceGroups/{1}/providers/Microsoft.Network/virtualNetworks/{2}/subnets/{3}'.format(subscription_id, resource_group_name, virtual_network_name, name)