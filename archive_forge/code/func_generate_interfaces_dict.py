from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.version import (
from datetime import datetime
import time
def generate_interfaces_dict(array):
    api_version = array._list_available_rest_versions()
    int_info = {}
    ports = array.list_ports()
    for port in range(0, len(ports)):
        int_name = ports[port]['name']
        if ports[port]['wwn']:
            int_info[int_name] = ports[port]['wwn']
        if ports[port]['iqn']:
            int_info[int_name] = ports[port]['iqn']
        if NVME_API_VERSION in api_version:
            if ports[port]['nqn']:
                int_info[int_name] = ports[port]['nqn']
    return int_info