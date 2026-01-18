from __future__ import absolute_import, division, print_function
import json
import time
from ansible.module_utils.basic import AnsibleModule
def cycle_over(module, executable, name):
    """Inspect each container in a cycle in case some of them don't exist.

    Arguments:
        module {AnsibleModule} -- instance of AnsibleModule
        executable {string} -- binary to execute when inspecting containers
        name {list} -- list of containers names to inspect

    Returns:
        list of containers info, stdout as empty, stderr
    """
    inspection = []
    stderrs = []
    for container in name:
        command = [executable, 'container', 'inspect', container]
        rc, out, err = module.run_command(command)
        if rc != 0 and 'no such ' not in err:
            module.fail_json(msg='Unable to gather info for %s: %s' % (container, err))
        if rc == 0 and out:
            json_out = json.loads(out)
            if json_out:
                inspection += json_out
        stderrs.append(err)
    return (inspection, '', '\n'.join(stderrs))