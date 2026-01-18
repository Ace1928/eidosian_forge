from __future__ import absolute_import, division, print_function
import time
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from copy import deepcopy
from ansible.module_utils.common.dict_transformations import (
def ip_configuration_id(subscription_id, resource_group_name, network_interface_name, name):
    """Generate the id for a request routing rule in an application gateway"""
    return '/subscriptions/{0}/resourceGroups/{1}/providers/Microsoft.Network/networkInterfaces/{2}/ipConfigurations/{3}'.format(subscription_id, resource_group_name, network_interface_name, name)