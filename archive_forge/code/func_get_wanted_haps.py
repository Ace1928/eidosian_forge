from __future__ import absolute_import, division, print_function
from ansible_collections.purestorage.fusion.plugins.module_utils.operations import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
from ansible_collections.purestorage.fusion.plugins.module_utils.parsing import (
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible.module_utils.basic import AnsibleModule
def get_wanted_haps(module):
    """Return set of host access policies to assign"""
    if not module.params['host_access_policies']:
        return set()
    return set([hap.strip() for hap in module.params['host_access_policies']])