from __future__ import absolute_import, division, print_function
import base64
import random
import re
import time
from ansible.module_utils.basic import to_native, to_bytes
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import (AzureRMModuleBase,
def delete_nic(self, resource_group, name):
    self.log('Deleting network interface {0}'.format(name))
    self.results['actions'].append('Deleted network interface {0}'.format(name))
    try:
        poller = self.network_client.network_interfaces.begin_delete(resource_group, name)
    except Exception as exc:
        self.fail('Error deleting network interface {0} - {1}'.format(name, str(exc)))
    self.get_poller_result(poller)
    return True