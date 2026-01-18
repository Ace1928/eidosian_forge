from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible_collections.purestorage.fusion.plugins.module_utils.parsing import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
from ansible_collections.purestorage.fusion.plugins.module_utils.operations import (
from ansible_collections.purestorage.fusion.plugins.module_utils.snapshots import (
def get_pp(module, fusion):
    """Return Protection Policy or None"""
    pp_api_instance = purefusion.ProtectionPoliciesApi(fusion)
    try:
        return pp_api_instance.get_protection_policy(protection_policy_name=module.params['name'])
    except purefusion.rest.ApiException:
        return None