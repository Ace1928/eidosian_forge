from __future__ import absolute_import, division, print_function
import base64
import random
import re
import time
from ansible.module_utils.basic import to_native, to_bytes
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import (AzureRMModuleBase,
def get_vm(self):
    """
        Get the VM with expanded instanceView

        :return: VirtualMachine object
        """
    try:
        vm = self.compute_client.virtual_machines.get(self.resource_group, self.name, expand='instanceview')
        return vm
    except Exception as exc:
        self.fail('Error getting virtual machine {0} - {1}'.format(self.name, str(exc)))