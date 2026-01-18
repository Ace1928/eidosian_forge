from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
def get_secret_info(module, executable, show, name):
    command = [executable, 'secret', 'inspect']
    if show:
        command.append('--showsecret')
    if name:
        command.append(name)
    else:
        all_names = [executable, 'secret', 'ls', '-q']
        rc, out, err = module.run_command(all_names)
        name = out.split()
        if not name:
            return ([], out, err)
        command.extend(name)
    rc, out, err = module.run_command(command)
    if rc != 0 or 'no secret with name or id' in err:
        module.fail_json(msg='Unable to gather info for %s: %s' % (name or 'all secrets', err))
    if not out or json.loads(out) is None:
        return ([], out, err)
    return (json.loads(out), out, err)