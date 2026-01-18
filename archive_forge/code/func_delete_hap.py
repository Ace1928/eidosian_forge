from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
from ansible_collections.purestorage.fusion.plugins.module_utils.operations import (
def delete_hap(module, fusion):
    """Delete a Host Access Policy"""
    hap_api_instance = purefusion.HostAccessPoliciesApi(fusion)
    changed = True
    if not module.check_mode:
        op = hap_api_instance.delete_host_access_policy(host_access_policy_name=module.params['name'])
        await_operation(fusion, op)
    module.exit_json(changed=changed)