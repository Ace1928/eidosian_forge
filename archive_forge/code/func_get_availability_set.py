from __future__ import absolute_import, division, print_function
import base64
import random
import re
import time
from ansible.module_utils.basic import to_native, to_bytes
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import (AzureRMModuleBase,
def get_availability_set(self, resource_group, name):
    try:
        return self.compute_client.availability_sets.get(resource_group, name)
    except Exception as exc:
        self.fail('Error fetching availability set {0} - {1}'.format(name, str(exc)))