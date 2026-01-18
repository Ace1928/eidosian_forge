from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
def remove_check(module, check_id):
    """ removes a check using its id """
    consul_api = get_consul_api(module)
    if check_id in consul_api.agent.checks():
        consul_api.agent.check.deregister(check_id)
        module.exit_json(changed=True, id=check_id)
    module.exit_json(changed=False, id=check_id)