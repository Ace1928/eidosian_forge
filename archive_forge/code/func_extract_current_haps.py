from __future__ import absolute_import, division, print_function
from ansible_collections.purestorage.fusion.plugins.module_utils.operations import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
from ansible_collections.purestorage.fusion.plugins.module_utils.parsing import (
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible.module_utils.basic import AnsibleModule
def extract_current_haps(volume):
    """Return set of host access policies that volume currently has"""
    if not volume.host_access_policies:
        return set()
    return set([hap.name for hap in volume.host_access_policies])