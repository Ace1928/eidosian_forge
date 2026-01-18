from __future__ import absolute_import, division, print_function
import base64
import random
import re
import time
from ansible.module_utils.basic import to_native, to_bytes
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import (AzureRMModuleBase,
def delete_managed_disks(self, managed_disk_ids):
    for mdi in managed_disk_ids:
        try:
            poller = self.rm_client.resources.begin_delete_by_id(mdi, '2017-03-30')
            self.get_poller_result(poller)
        except Exception as exc:
            self.fail('Error deleting managed disk {0} - {1}'.format(mdi, str(exc)))
    return True