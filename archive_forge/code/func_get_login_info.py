from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
def get_login_info(module, executable, authfile, registry):
    command = [executable, 'login', '--get-login']
    result = dict(registry=registry, username='', logged_in=False)
    if authfile:
        command.extend(['--authfile', authfile])
    if registry:
        command.append(registry)
    rc, out, err = module.run_command(command)
    if rc != 0:
        if 'Error: not logged into' in err:
            result['registry'] = err.split()[-1]
            err = ''
            return result
        module.fail_json(msg='Unable to gather info for %s: %s' % (registry, err))
    result['username'] = out.strip()
    result['logged_in'] = True
    return result