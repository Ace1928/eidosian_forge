from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
def get_pod_info(module, executable, name):
    command = [executable, 'pod', 'inspect']
    pods = [name]
    result = []
    errs = []
    rcs = []
    if not name:
        all_names = [executable, 'pod', 'ls', '-q']
        rc, out, err = module.run_command(all_names)
        if rc != 0:
            module.fail_json(msg='Unable to get list of pods: %s' % err)
        name = out.split()
        if not name:
            return ([], [err], [rc])
        pods = name
    for pod in pods:
        rc, out, err = module.run_command(command + [pod])
        errs.append(err.strip())
        rcs += [rc]
        if not out or json.loads(out) is None or (not json.loads(out)):
            continue
        result.append(json.loads(out))
    return (result, errs, rcs)