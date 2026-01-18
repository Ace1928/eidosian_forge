from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble import __version__ as NIMBLE_ANSIBLE_VERSION
import ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble as utils
def delete_perf_policy(client_obj, perf_policy_name):
    if utils.is_null_or_empty(perf_policy_name):
        return (False, False, 'Delete performance policy failed. Performance policy name is not present.', {})
    try:
        perf_policy_resp = client_obj.performance_policies.get(id=None, name=perf_policy_name)
        if utils.is_null_or_empty(perf_policy_resp):
            return (False, False, f"Cannot delete Performance policy '{perf_policy_name}' as it is not present ", {})
        else:
            perf_policy_resp = client_obj.performance_policies.delete(id=perf_policy_resp.attrs.get('id'))
            return (True, True, f"Deleted performance policy '{perf_policy_name}' successfully.", {})
    except Exception as ex:
        return (False, False, f'Performance policy deletion failed | {ex}', {})