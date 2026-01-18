from __future__ import absolute_import, division, print_function
import base64
import random
import re
import time
from ansible.module_utils.basic import to_native, to_bytes
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import (AzureRMModuleBase,
def get_network_interface(self, resource_group, name):
    try:
        nic = self.network_client.network_interfaces.get(resource_group, name)
        return nic
    except ResourceNotFoundError as exc:
        self.fail('Error fetching network interface {0} - {1}'.format(name, str(exc)))
    return True