from __future__ import absolute_import, division, print_function
import re
import time
import string
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.network import is_mac
from ansible.module_utils._text import to_text, to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible_collections.community.vmware.plugins.module_utils.vm_device_helper import PyVmomiDeviceHelper
from ansible_collections.community.vmware.plugins.module_utils.vmware_spbm import SPBM
def autoselect_datastore(self):
    datastore = None
    datastores = self.cache.get_all_objs(self.content, [vim.Datastore])
    if datastores is None or len(datastores) == 0:
        self.module.fail_json(msg='Unable to find a datastore list when autoselecting')
    datastore_freespace = 0
    for ds in datastores:
        if not self.is_datastore_valid(datastore_obj=ds):
            continue
        if ds.summary.freeSpace > datastore_freespace:
            datastore = ds
            datastore_freespace = ds.summary.freeSpace
    return datastore