from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
def podmanExec(module, target, filters, executable):
    command = [executable, target, 'prune', '--force']
    if filters is not None:
        command.extend(filtersPrepare(target, filters))
    rc, out, err = module.run_command(command)
    changed = bool(out)
    if rc != 0:
        module.fail_json(msg='Error executing prune on {target}: {err}'.format(target=target, err=err))
    return {'changed': changed, target: list(filter(None, out.split('\n'))), 'errors': err}