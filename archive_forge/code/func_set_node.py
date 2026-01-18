from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
def set_node(module, state, timeout, force, node='all'):
    if state == 'online':
        cmd = 'pcs cluster start'
    if state == 'offline':
        cmd = 'pcs cluster stop'
        if force:
            cmd = '%s --force' % cmd
    nodes_state = get_node_status(module, node)
    for node in nodes_state:
        if node[1].strip().lower() != state:
            cmd = '%s %s' % (cmd, node[0].strip())
            rc, out, err = module.run_command(cmd)
            if rc == 1:
                module.fail_json(msg='Command execution failed.\nCommand: `%s`\nError: %s' % (cmd, err))
    t = time.time()
    ready = False
    while time.time() < t + timeout:
        nodes_state = get_node_status(module)
        for node in nodes_state:
            if node[1].strip().lower() == state:
                ready = True
                break
    if not ready:
        module.fail_json(msg='Failed to set the state `%s` on the cluster\n' % state)