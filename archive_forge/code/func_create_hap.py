from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
from ansible_collections.purestorage.fusion.plugins.module_utils.operations import (
def create_hap(module, fusion):
    """Create a new host access policy"""
    hap_api_instance = purefusion.HostAccessPoliciesApi(fusion)
    changed = True
    if not module.check_mode:
        display_name = module.params['display_name'] or module.params['name']
        op = hap_api_instance.create_host_access_policy(purefusion.HostAccessPoliciesPost(iqn=module.params['iqn'], personality=module.params['personality'], name=module.params['name'], display_name=display_name))
        res_op = await_operation(fusion, op)
        id = res_op.result.resource.id
    module.exit_json(changed=changed, id=id)