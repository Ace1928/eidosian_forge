from __future__ import absolute_import, division, print_function
import base64
import random
import re
import time
from ansible.module_utils.basic import to_native, to_bytes
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import (AzureRMModuleBase,
def create_or_update_vm(self, params, remove_autocreated_on_failure):
    try:
        poller = self.compute_client.virtual_machines.begin_create_or_update(self.resource_group, self.name, params)
        self.get_poller_result(poller)
    except Exception as exc:
        if remove_autocreated_on_failure:
            self.remove_autocreated_resources(params.tags)
        self.fail('Error creating or updating virtual machine {0} - {1}'.format(self.name, str(exc)))