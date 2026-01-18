from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.argspec.facts.facts import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.facts import Facts
def get_chassis_type(connection):
    """Return facts resource subsets based on
    chassis model.
    """
    target_type = 'nexus'
    device_info = connection.get_device_info()
    model = device_info.get('network_os_model', '')
    platform = device_info.get('network_os_platform', '')
    if platform.startswith('DS-') and 'MDS' in model:
        target_type = 'mds'
    return target_type