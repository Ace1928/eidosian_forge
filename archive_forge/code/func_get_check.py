from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils import arguments, errors, utils
def get_check(client, namespace, check):
    check_path = utils.build_core_v2_path(namespace, 'checks', check)
    resp = client.get(check_path)
    if resp.status != 200:
        raise errors.SyncError("Check with name '{0}' does not exist on remote.".format(check))
    return resp.json