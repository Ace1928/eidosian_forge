from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble import __version__ as NIMBLE_ANSIBLE_VERSION
import ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble as utils
def create_perf_policy(client_obj, perf_policy_name, **kwargs):
    if utils.is_null_or_empty(perf_policy_name):
        return (False, False, 'Create performance policy failed. Performance policy name is not present.', {}, {})
    try:
        perf_policy_resp = client_obj.performance_policies.get(id=None, name=perf_policy_name)
        if utils.is_null_or_empty(perf_policy_resp):
            params = utils.remove_null_args(**kwargs)
            perf_policy_resp = client_obj.performance_policies.create(name=perf_policy_name, **params)
            if perf_policy_resp is not None:
                return (True, True, f"Created performance policy '{perf_policy_name}' successfully.", {}, perf_policy_resp.attrs)
        else:
            return (False, False, f"Cannot create Performance policy '{perf_policy_name}' as it is already present", {}, {})
    except Exception as ex:
        return (False, False, f'Performance policy creation failed | {ex}', {}, {})