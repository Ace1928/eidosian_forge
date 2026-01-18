from __future__ import absolute_import, division, print_function
import platform
import re
from ansible.module_utils.six import iteritems
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.utils.utils import (
def ipv6_structure_op_supported(self):
    data = self.capabilities
    if data:
        nxos_os_version = data['device_info']['network_os_version']
        unsupported_versions = ['I2', 'F1', 'A8']
        for ver in unsupported_versions:
            if ver in nxos_os_version:
                return False
        return True