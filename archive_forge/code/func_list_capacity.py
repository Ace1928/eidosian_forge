from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.version import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
def list_capacity(module, array):
    """Get avaible expansion points"""
    steps = list(array.get_arrays_cloud_capacity_supported_steps().items)
    available = []
    for step in range(0, len(steps)):
        available.append(steps[step].supported_capacity)
    module.exit_json(changed=True, available=available)