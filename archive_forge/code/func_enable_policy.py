from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
def enable_policy(module, names, act):
    policies = []
    for name in names:
        if not is_policy_enabled(module, name):
            policies.append(name)
    if not policies:
        module.exit_json(changed=False, msg='policy(ies) already enabled')
    names = ' '.join(policies)
    if module.check_mode:
        cmd = '%s list' % AWALL_PATH
    else:
        cmd = '%s enable %s' % (AWALL_PATH, names)
    rc, stdout, stderr = module.run_command(cmd)
    if rc != 0:
        module.fail_json(msg='failed to enable %s' % names, stdout=stdout, stderr=stderr)
    if act and (not module.check_mode):
        activate(module)
    module.exit_json(changed=True, msg='enabled awall policy(ies): %s' % names)