from __future__ import absolute_import, division, print_function
import base64
import random
import re
import time
from ansible.module_utils.basic import to_native, to_bytes
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import (AzureRMModuleBase,
def power_on_vm(self):
    self.results['actions'].append('Powered on virtual machine {0}'.format(self.name))
    self.log('Power on virtual machine {0}'.format(self.name))
    try:
        poller = self.compute_client.virtual_machines.begin_start(self.resource_group, self.name)
        self.get_poller_result(poller)
    except Exception as exc:
        self.fail('Error powering on virtual machine {0} - {1}'.format(self.name, str(exc)))
    return True